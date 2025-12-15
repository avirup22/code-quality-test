import asyncio
import base64
import hashlib
from pathlib import Path
import httpx
import io
import json
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import List, Optional, Dict, Any

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector

import anyio
import boto3
from botocore.exceptions import ClientError
import fal_client
import requests
from botocore.config import Config
from fastapi import status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from loguru import logger
from openai import AsyncOpenAI, OpenAIError
from PIL import Image as PILImage
from pymongo import AsyncMongoClient
from redis.asyncio import Redis
from sqlalchemy import delete, func, or_, select, and_, update, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, aliased
from sqlalchemy.orm.attributes import flag_modified
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_fixed)
from typing import Set

from app.core.config import settings
from app.api.deps import get_project_code
from app.database.db import create_async_engine, SessionLocal
from app.models import (CompositionReference, Country, Image, ImageType, GeographicEntities,
                        ModelDimensions, Product, Project, Prompt, LicenseFile,
                        StyleReference)
from app.models.images import ImageStatus, ImageSource
from app.models.images import ImageType as Type

from app.schemas import ResponseCode, ResponseStatus, ResponseType, SSEResponse

from .prompts import (content_enrich_prompt, core_intent_extractor,
                      edit_description, ethnicity_enrich_prompt,polish_prompt,
                      extract_description, guardrail_presets, extract_icon_description, edit_icon_description,
                      guardrail_to_preset, negative_prompt_generator,
                      non_human_enrich_prompt, product_enrich_prompt,
                      product_prompt, refine_prompt, restructure_prompt)
from .utils import (
    analyze_image, check_compliance, copy_s3_object, delete_from_s3, format_date,
    generate_presigned_url, generate_token, get_root_and_parent, generate_next_image_name,
    get_image, prepare_prompt_and_model,
    incorporate_geography_nuances, build_cloudfront_url,
    preset_generator, process_and_save_images, get_next_iteration,
    upload_to_s3, format_version, extract_values, calculate_compliance_ratio, get_composition_reference_url, get_style_reference_url)

async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
redis = Redis.from_url(settings.REDIS_URL)

THUMBNAIL_MAX_SIZE = (512, 512)


class ImageGen:

    def __init__(self):
        self.background_tasks: Set[asyncio.Task] = set()

    async def get_filters(self, db: AsyncSession):

        logger.info("Starting get_filters operation")
        try:
            data = {}

            logger.debug("Fetching products from database")
            result = await db.execute(select(Product).order_by(Product.value.asc()))
            products = result.scalars().all()
            logger.info(f"Successfully fetched {len(products)} products")

            data["product"] = [
                {"key": product.key, "value": product.value}
                for product in products
            ]

            logger.debug("Fetching countries from database")
            result = await db.execute(select(Country).order_by(Country.value.asc()))
            countries = result.scalars().all()
            logger.info(f"Successfully fetched {len(countries)} countries")

            data["country"] = [
                {"key": country.key, "value": country.value}
                for country in countries
            ]

            logger.debug("Fetching image types from database")
            result = await db.execute(select(ImageType).order_by(ImageType.value.asc()))
            types = result.scalars().all()
            logger.info(f"Successfully fetched {len(types)} image types")

            data["type"] = [
                {
                    "key": image_type.key,
                    "value": image_type.value
                }
                for image_type in types
            ]

            logger.info("get_filters operation completed successfully")
            return data

        except Exception as e:
            logger.error(f"Exception occurred in get_filters: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=settings.err_msg
            )

    async def get_image_dimensions(self, db: AsyncSession):

        logger.info("Starting get_image_dimensions operation")
        try:
            logger.debug("Fetching model dimensions for model: adobe")
            result = await db.execute(
                select(ModelDimensions)
                .filter(ModelDimensions.model == "adobe")
                .order_by(ModelDimensions.sort_order)
            )

            dimensions = result.scalars().all()
            logger.info(
                f"Successfully fetched {len(dimensions)} model dimensions for adobe")

            data = [
                {
                    "id": item.id,
                    "name": item.name,
                    "width": item.width,
                    "height": item.height,
                    "aspect_ratio": item.aspect_ratio,
                }
                for item in dimensions
            ]

            logger.info(
                "get_image_dimensions operation completed successfully")
            return data

        except Exception as e:
            logger.error(
                f"Exception occurred in get_image_dimensions: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=settings.err_msg
            )

    async def get_style_reference(self, db: AsyncSession):

        logger.info("Starting get_style_reference operation")

        try:
            logger.debug("Fetching style references from database")
            result = await db.execute(
                select(StyleReference).order_by(StyleReference.id)
            )
            query = result.scalars().all()
            logger.info(f"Successfully fetched {len(query)} style references")

            if query:
                logger.debug("Generating presigned URLs for style references")
                records = []
                encoded_query = jsonable_encoder(query)

                for record in encoded_query:
                    presigned_url = build_cloudfront_url(
                        record["s3_object_key"])
                    records.append({"id": record["id"], "url": presigned_url})

                logger.info(
                    f"Successfully generated {len(records)} presigned URLs")
                return records
            else:
                logger.warning("No style references found in database")
                return []

        except Exception as e:
            logger.error(
                f"Exception occurred in get_style_reference: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=settings.err_msg
            )

    async def get_composition_reference(self, db: AsyncSession):

        logger.info("Starting get_composition_reference operation")

        try:
            logger.debug("Fetching composition references from database")
            result = await db.execute(
                select(CompositionReference).order_by(CompositionReference.id)
            )
            query = result.scalars().all()
            logger.info(
                f"Successfully fetched {len(query)} composition references")

            if query:
                logger.debug(
                    "Generating presigned URLs for composition references")
                records = []
                encoded_query = jsonable_encoder(query)

                for record in encoded_query:
                    presigned_url = generate_presigned_url(
                        record["s3_object_key"])
                    records.append({"id": record["id"], "url": presigned_url})

                logger.info(
                    f"Successfully generated {len(records)} presigned URLs")
                return records
            else:
                logger.warning("No composition references found in database")
                return []
        except Exception as e:
            logger.error(
                f"Exception occurred in get_composition_reference: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=settings.err_msg
            )

    async def create_reference(self, db: AsyncSession, record, file, type):
        logger.info(f"Starting create_reference operation for type: {type}")
        try:
            reference_model = StyleReference if type == "style" else CompositionReference

            if record:
                original_file_name = record.s3_object_key.split("/")[-1]
                # Add UUID to filename from existing record
                file_path = Path(original_file_name)
                file_stem = file_path.stem
                file_ext = file_path.suffix
                unique_id = uuid.uuid4().hex[:8]
                file_name = f"{file_stem}_{unique_id}{file_ext}"

                src_key = record.s3_object_key
                dest_key = f"gilead/imagegen/{type}reference/{file_name}"
                logger.info(
                    f"Preparing to copy existing S3 object from {src_key} to {dest_key}")
            elif file:
                # Add UUID to uploaded filename
                file_path = Path(file.filename)
                file_stem = file_path.stem
                file_ext = file_path.suffix
                unique_id = uuid.uuid4().hex[:8]
                file_name = f"{file_stem}_{unique_id}{file_ext}"

                dest_key = f"gilead/imagegen/{type}reference/{file_name}"
                logger.info(
                    f"Preparing to upload new file: {file_name} to {dest_key}")
            else:
                logger.warning(
                    "Neither file nor record provided for create_reference")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either file or record must be provided."
                )

            # The duplicate check is now less likely to trigger, but keep it as a safety measure
            logger.debug(
                f"Checking if {type} reference already exists at {dest_key}")
            existing_result = await db.execute(
                select(reference_model).filter_by(s3_object_key=dest_key)
            )
            existing_reference = existing_result.scalars().first()

            if existing_reference:
                logger.warning(
                    f"{type.capitalize()} reference already exists at {dest_key}")
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"{type.capitalize()} reference already exists."
                )

            if record:
                logger.debug(f"Copying S3 object from {src_key} to {dest_key}")
                copy_s3_object(src_key, dest_key)
                logger.info(f"Successfully copied S3 object to {dest_key}")
            else:
                file_content = file.file.read()
                file_size_kb = len(file_content) / 1024
                logger.debug(
                    f"Uploading file to S3: {dest_key}, size: {file_size_kb:.2f} KB, content_type: {file.content_type}")
                upload_to_s3(
                    file=BytesIO(file_content),
                    file_key=dest_key,
                    content_type=file.content_type or "image/jpeg"
                )
                logger.info(f"Successfully uploaded file to {dest_key}")

            logger.debug(f"Creating database entry for {type} reference")
            db_obj = reference_model(
                style_name=file_name,  # Store the unique filename
                s3_object_key=dest_key
            )
            db.add(db_obj)
            await db.commit()
            await db.refresh(db_obj)
            logger.info(f"Database entry created with id: {db_obj.id}")

            cache_key = f"{type}_reference_cache"
            logger.debug(f"Invalidating cache key: {cache_key}")
            await redis.delete(cache_key)
            logger.debug(f"Cache invalidated for {cache_key}")

            logger.debug(f"Generating presigned URL for {dest_key}")
            presigned_url = generate_presigned_url(dest_key)

            logger.info(
                f"{type.capitalize()} reference created successfully with id: {db_obj.id}")
            return {"id": db_obj.id, "url": presigned_url}

        except HTTPException as http_err:
            logger.warning(
                f"HTTPException in create_reference: {http_err.status_code} - {http_err.detail}")
            raise
        except Exception as e:
            logger.error(
                f"Exception occurred in create_reference for type {type}: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=settings.err_msg
            )

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(3),
        retry=retry_if_exception_type((OpenAIError, Exception)),
        reraise=True,
    )
    async def get_image_analysis(self, image_obj, image_type, prompt_obj):
        try:
            logger.info(
                f"Starting get_image_analysis for image_id={getattr(image_obj, 'id', None)}, prompt_id={getattr(prompt_obj, 'id', None)}")
            logger.debug(f"Image object: {image_obj}")
            logger.debug(f"Prompt object: {prompt_obj}")

            s3_object_key = image_obj.s3_object_key
            logger.info(
                f"Generating presigned URL for S3 key: {s3_object_key}")
            image_url = generate_presigned_url(s3_object_key)
            logger.info(f"Image URL generated: {image_url}")

            prompt = getattr(prompt_obj, 'enhanced_prompt', None)
            logger.info(f"Using enhanced prompt: {prompt}")

            preset_dict = getattr(prompt_obj, 'guidelines_applied', None)
            logger.debug(f"Guidelines applied (preset_dict): {preset_dict}")
            presets = extract_values(preset_dict)
            logger.info(f"Extracted presets for guardrails: {presets}")

            logger.info("Calling analyze_image API...")
            data = await analyze_image(
                image_url=image_url,
                presets=presets,
                prompt=prompt,
            )
            logger.info(
                f"Image analysis completed. Matched guardrails: {data.get('matched_guardrails')}, Missing guardrails: {data.get('missing_guardrails')}")
            logger.debug(f"Analysis API response: {data}")

            applied_guardrails = check_compliance(
                preset_dict=preset_dict,
                matched_guardrails=data["matched_guardrails"],
                missing_guardrails=data["missing_guardrails"],
            )
            logger.info(
                f"Compliance checked. Applied guardrails: {applied_guardrails}")

            compliance_score = calculate_compliance_ratio(
                data["matched_guardrails"], data["missing_guardrails"])
            logger.info(f"Compliance score calculated: {compliance_score}")

            result = {
                "id": image_obj.id,
                "is_published": image_obj.is_published,
                "image_url": image_url,
                "prompt": prompt_obj.prompt,
                "analysis": {
                    "score": str(compliance_score),
                    "suggestion": data["suggestion"],
                    "suggestedPrompt": data["suggested_prompt"],
                    "guardrails": applied_guardrails,
                },
            }
            logger.info(
                f"Returning image analysis result for image_id={image_obj.id}")
            logger.debug(f"Image analysis result: {result}")
            return result
        except Exception as e:
            logger.error(
                f"Exception in get_image_analysis for image_id={getattr(image_obj, 'id', None)}: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=settings.err_msg
            )

    async def generate_image(self, db: AsyncSession, data: dict, prompt: str, parameters: dict, project_id=None):
        try:
            image_type = parameters.get("type")
            enhanced_prompt, preset_dict, model, content_class, negative_prompt = await prepare_prompt_and_model(prompt, parameters)
            prompt_obj = Prompt(
                prompt=prompt, enhanced_prompt=enhanced_prompt, guidelines_applied=preset_dict)
            db.add(prompt_obj)
            await db.flush()
            # Get Adobe Access Token
            try:
                access_token = await generate_token()
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to retrieve access token: {str(e)}"
                )

            if not access_token:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Access token is invalid."
                )

            # --- Prepare Style/Composition URLs ---
            style_url = None
            if data.get("style_reference_id"):
                result = await db.execute(select(StyleReference).where(
                    StyleReference.id == data["style_reference_id"]
                ))
                style_ref = result.scalar_one_or_none()
                if style_ref:
                    style_url = generate_presigned_url(style_ref.s3_object_key)

            composition_url = None
            if data.get("composition_reference_id"):
                result = await db.execute(select(CompositionReference).where(
                    CompositionReference.id == data["composition_reference_id"]
                ))
                comp_ref = result.scalar_one_or_none()
                if comp_ref:
                    composition_url = generate_presigned_url(
                        comp_ref.s3_object_key)

            # --- Firefly API Payload ---
            payload = {
                "contentClass": content_class,
                "negativePrompt": negative_prompt,
                "numVariations": data["number_of_variations"],
                "prompt": enhanced_prompt[:1024],
                "size": {"height": data["height"], "width": data["width"]},
                "visualIntensity": 5
            }
            if style_url:
                payload["style"] = {"imageReference": {
                    "source": {"url": style_url}}, "strength": 70}
            if composition_url:
                payload["structure"] = {"imageReference": {
                    "source": {"url": composition_url}}, "strength": 70}
            if image_type == "icon":
                payload["customModelId"] = settings.ADOBE_CUSTOM_MODEL
                payload.pop("visualIntensity")
            # --- Firefly API Request ---
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": settings.ADOBE_CLIENT_ID,
                "Authorization": f"Bearer {access_token}",
                "x-model-version": model
            }
            logger.info(f"Payload for image generation: {payload}")
            response = requests.post(
                "https://firefly-api.adobe.io/v3/images/generate-async",
                headers=headers,
                data=json.dumps(payload),
                timeout=90
            )
            if response.status_code != 202:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Image generation failed: {response.text}"
                )
            status_url = response.json()["statusUrl"]

            # --- Poll for Firefly results ---
            max_retries, retry_delay = 60, 2
            image_urls = []
            for _ in range(max_retries):
                status_res = get_image(status_url, access_token)
                status_val = status_res["status"]
                if status_val == "succeeded":
                    image_urls = status_res["result"]["outputs"]
                    break
                elif status_val == "failed":
                    raise HTTPException(
                        status_code=500, detail=str(status_res))
                await asyncio.sleep(retry_delay)
            else:
                raise HTTPException(
                    status_code=504, detail="Image generation timed out.")

            # --- Process & upload images to S3 ---
            print(image_urls)
            batch_id = uuid.uuid4()
            next_iteration = await get_next_iteration(db, project_id)
            saved_images = await process_and_save_images(db, image_urls, data,
                                                         prompt_obj.id, project_id, batch_id, next_iteration, preset_dict)
            await db.commit()

            try:
                await self._generate_prompt_embedding(
                    prompt_id=prompt_obj.id,
                    prompt=prompt,
                    enhanced_prompt=enhanced_prompt,
                    project_id=project_id
                )
            except Exception as embed_error:
                logger.error(
                    f"Failed to generate embedding: {str(embed_error)}")
                logger.error(traceback.format_exc())

            return {
                "status": "success",
                "data": {
                    "iteration": next_iteration,
                    "iteration_type": "generation",
                    "project_id": project_id,
                    "prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "images": saved_images,
                    "variationrootimage": None
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Unhandled exception in generate_image:")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=settings.err_msg
            )

    async def _generate_prompt_embedding(
        self,
        prompt_id: int,
        prompt: str,
        enhanced_prompt: str,
        project_id: int = None
    ):
        try:
            logger.info(
                f"Starting embedding generation for prompt {prompt_id}")

            async with SessionLocal() as session:
                text_to_embed = f"{prompt.strip() if prompt else ''}\n\nEnhanced: {enhanced_prompt.strip() if enhanced_prompt else ''}".strip()

                if not text_to_embed:
                    logger.warning(
                        f"Empty text skipped for prompt {prompt_id}")
                    return  # Just return, don't raise HTTPException

                brand = geography = classification = None
                if project_id:
                    result = await session.execute(
                        select(Project).where(Project.id == project_id)
                    )
                    project = result.scalar_one_or_none()
                    if project:
                        brand = project.product.lower().strip() if project.product else None
                        geography = project.country.lower().strip() if project.country else None
                        classification = project.type.lower().strip() if project.type else None

                metadata = {
                    "prompt_id": str(prompt_id),
                    "project_id": str(project_id) if project_id else None,
                    "type": "image_generation_prompt",
                    "brand": brand,
                    "geography": geography,
                    "classification": classification
                }

                logger.info(
                    f"[BG] Generating embedding for prompt {prompt_id}")

                embed_model = OpenAIEmbeddings(model=settings.EMBED_MODEL)
                vectorstore = PGVector(
                    connection_string=settings.SQLALCHEMY_DATABASE_SSL_URI,
                    embedding_function=embed_model,
                    collection_name="local_prompts"
                )
                vectorstore.add_texts(
                    texts=[text_to_embed],
                    metadatas=[metadata]
                )
                await session.commit()
                logger.info(f"[BG] Embedding stored for prompt {prompt_id}")
        except Exception as e:
            logger.error(
                f"Failed to store embedding for {prompt_id}: {str(e)}")
            logger.error(traceback.format_exc())

    async def generate_image_variations(self, db, number_of_variations, image):
        try:
            logger.info("Starting image generation with Adobe Firefly API.")
            try:
                access_token = await generate_token()
            except Exception as e:
                logger.error(f"Failed to retrieve access token: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve access token.",
                )
            if not access_token:
                logger.error("Access token is None or empty.")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Access token is invalid.",
                )
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": settings.ADOBE_CLIENT_ID,
                "Authorization": f"Bearer {access_token}",
                "x-model-version": "image4_ultra"
            }
            url = generate_presigned_url(image.s3_object_key)
            payload = {
                "image": {"source": {"url": url}},
                "numVariations": number_of_variations,
                "size": {"height": image.height, "width": image.width}
            }
            logger.info(f"Payload for Adobe Firefly API: {payload}")
            response = requests.post(
                "https://firefly-api.adobe.io/v3/images/generate-similar-async",
                headers=headers,
                data=json.dumps(payload),
            )
            if response.status_code != 202:
                logger.error("Failed to generate image: %s", response.text)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=response.text
                )
            res = response.json()
            status_url = res["statusUrl"]
            logger.info("Starting polling for image generation status...")
            max_retries = 60
            retry_delay = 2
            image_urls = []
            for _ in range(max_retries):
                image_fetch_res = get_image(status_url, access_token)
                status_value = image_fetch_res.get("status")
                logger.info("Firefly status: %s", status_value)
                if status_value == "succeeded":
                    image_urls = image_fetch_res["result"]["outputs"]
                    break
                elif status_value == "failed":
                    logger.error("Image generation failed: %s",
                                 image_fetch_res)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(image_fetch_res)
                    )
                elif status_value in ("pending", "running"):
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Unexpected status: %s", status_value)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Unexpected status: {status_value}"
                    )
            else:
                logger.error(
                    "Image generation timed out after %d retries", max_retries)
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Timed out."
                )
            logger.info("Image generation succeeded, returning results.")
            batch_id = uuid.uuid4()

            prompt_record = await db.execute(
                select(Prompt).where(Prompt.id == image.prompt_id)
            )
            result = prompt_record.scalar_one_or_none()

            if result and result.guidelines_applied:
                validate = True
            else:
                validate = False

            # UPDATED: Simplified lineage context - variations don't inherit parent lineage
            lineage_context = get_root_and_parent(image)
            version_source_id = image.id

            logger.info(f"Generating variations with context: parent_id={image.id}, "
                        f"parent_type={image.image_type}, parent_version={image.lineage_version_number}")

            saved_images = []
            for idx, img_data in enumerate(image_urls):
                try:
                    url = img_data.get("image", {}).get("url")
                    seed_value = img_data.get("seed")
                    if not url:
                        logger.warning(
                            f"[{idx}] Missing URL for image data: {img_data}")
                        continue
                    resp = await asyncio.to_thread(requests.get, url, timeout=10)
                    if resp.status_code != 200:
                        logger.warning(
                            f"[{idx}] Image download failed ({resp.status_code}): {url}")
                        continue
                    image_bytes = resp.content
                    uid = uuid.uuid4().hex[:8]
                    file_key = f"gilead/imagegen/generatedvariations/originals/image_{uid}.jpg"
                    thumb_key = f"gilead/imagegen/generatedvariations/thumbnails/thumb_{uid}.webp"

                    # Upload full image
                    upload_to_s3(io.BytesIO(image_bytes), file_key,
                                 content_type="image/jpeg")

                    # Generate and upload thumbnail
                    img = PILImage.open(io.BytesIO(image_bytes))
                    img.thumbnail(THUMBNAIL_MAX_SIZE)
                    thumb_buffer = io.BytesIO()
                    img.save(thumb_buffer, format="WEBP",
                             optimize=True, quality=100)
                    thumb_buffer.seek(0)
                    upload_to_s3(thumb_buffer, thumb_key,
                                 content_type="image/webp")
                    image_name = await generate_next_image_name(db)

                    # UPDATED: Variations are ALWAYS v1 of their own independent lineage
                    img_obj = Image(
                        user_id="system",
                        width=str(image.width),
                        height=str(image.height),
                        image_name=image_name,
                        s3_object_key=file_key,
                        s3_thumbnail_key=thumb_key,
                        prompt_id=image.prompt_id,
                        project_id=image.project_id,
                        style_reference_id=image.style_reference_id,
                        composition_reference_id=image.composition_reference_id,
                        seed=seed_value,
                        status=ImageStatus.TEMPORARY.value,
                        image_type=Type.VARIATION.value,
                        image_source=ImageSource.LOCAL.value,
                        root_id=lineage_context['root_id'],
                        parent_id=lineage_context['parent_id'],
                        version_source_id=version_source_id,
                        version_number=image.version_number,
                        iteration_number=None,
                        lineage_root_id=None,
                        lineage_version_number=1,
                        is_base=True,
                        generation_batch_id=batch_id,
                    )
                    db.add(img_obj)
                    await db.flush()

                    # CRITICAL: Set lineage_root_id to self (variation becomes its own root)
                    img_obj.lineage_root_id = img_obj.id
                    await db.flush()

                    logger.info(f"Created variation {img_obj.image_name}: "
                                f"lineage_root_id={img_obj.id} (self), lineage_version=1")

                    saved_images.append({
                        "id": str(img_obj.id),
                        "display_id": img_obj.image_name,
                        "url": generate_presigned_url(file_key),
                        "thumbnail_url": generate_presigned_url(thumb_key),
                        "is_published": img_obj.is_published,
                        "has_history": False,
                        "can_validate": validate,
                        "source": img_obj.image_source.lower()
                    })
                except Exception as e:
                    logger.error(f"[{idx}] Failed to process image: {e}")
                    continue
            await db.commit()
            logger.info(
                f"Successfully generated {len(saved_images)} variations")
            return saved_images

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Exception during image variations: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Image variations failed."
            )

    async def save_image_variations(self, db, image_ids: List[str]):
        try:
            logger.info(f"Saving {len(image_ids)} variations")
            unique_ids = list(set(image_ids))
            logger.info(f"Processing {len(unique_ids)} unique image IDs")

            result = await db.execute(
                select(Image).where(Image.id.in_(unique_ids))
            )
            images = result.scalars().all()

            if not images:
                logger.error("No images found with the provided IDs.")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No images found."
                )

            project_id = images[0].project_id

            found_ids = {str(img.id) for img in images}
            missing_ids = set(unique_ids) - found_ids
            if missing_ids:
                logger.warning(f"Missing image IDs: {missing_ids}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Some images not found: {', '.join(list(missing_ids)[:5])}"
                )

            batch_ids = {
                img.generation_batch_id for img in images if img.generation_batch_id}
            logger.info(
                f"Variations are from {len(batch_ids)} different batches")

            # Get parent image (all variations come from same parent)
            parent_image = None
            parent_prompt_id = None
            validate = False

            if images and images[0].parent_id:
                parent_result = await db.execute(
                    select(Image).where(Image.id == images[0].parent_id)
                )
                parent_image = parent_result.scalar_one_or_none()

                if parent_image:
                    logger.info(f"Parent image: {parent_image.image_name}")
                    parent_prompt_id = parent_image.prompt_id

                    # Check if parent has prompt and validate preset_dict
                    if parent_prompt_id:
                        prompt_record = await db.execute(
                            select(Prompt).where(Prompt.id == parent_prompt_id)
                        )
                        prompt_result = prompt_record.scalar_one_or_none()

                        if prompt_result and prompt_result.guidelines_applied:
                            validate = True
                            logger.info(
                                f"Parent prompt has preset_dict - validation enabled")
                        else:
                            validate = False
                            logger.info(
                                f"Parent prompt has no preset_dict - validation disabled")
                    else:
                        logger.info(f"Parent image has no prompt_id")
                else:
                    logger.warning(f"Parent image not found")

            # Create shared timestamp and batch ID for all saved variations
            shared_timestamp = datetime.now(timezone.utc)
            shared_batch_id = uuid.uuid4()
            max_iteration_result = await db.execute(
                select(func.max(Image.iteration_number))
                .where(Image.project_id == project_id)
            )
            max_iteration = max_iteration_result.scalar()
            next_iteration = (max_iteration or 0) + 1
            logger.info(
                f"Creating new batch for saved variations: {shared_batch_id}")

            saved_images = []
            saved_image_ids = []

            for image in images:
                try:
                    if image.status != ImageStatus.TEMPORARY.value:
                        logger.warning(
                            f"Image {image.id} is not temporary. Skipping.")
                        continue
                    if image.image_type != Type.VARIATION.value:
                        logger.warning(
                            f"Image {image.id} is not a variation. Skipping.")
                        continue

                    # Update status and lineage fields
                    image.status = ImageStatus.PERMANENT.value
                    image.is_base = True
                    image.lineage_version_number = 1
                    image.iteration_number = next_iteration

                    # Associate variation with parent's prompt_id
                    if parent_prompt_id:
                        image.prompt_id = parent_prompt_id
                        logger.info(
                            f"Associated {image.image_name} with parent prompt_id: {parent_prompt_id}")

                    # Update with shared timestamp and batch ID
                    image.created_at = shared_timestamp
                    image.generation_batch_id = shared_batch_id

                    # Safety check: Ensure lineage_root_id is set to self
                    if not image.lineage_root_id or image.lineage_root_id != image.id:
                        image.lineage_root_id = image.id
                        logger.warning(
                            f"Fixed lineage_root_id for {image.image_name}")

                    saved_image_ids.append(image.id)

                    # Generate presigned URLs
                    url = generate_presigned_url(image.s3_object_key)
                    thumbnail_url = generate_presigned_url(
                        image.s3_thumbnail_key)

                    saved_images.append({
                        "id": str(image.id),
                        "display_id": image.image_name,
                        "url": url,
                        "thumbnail_url": thumbnail_url,
                        "is_published": image.is_published,
                        "has_history": False,
                        "can_validate": validate,
                        "source": image.image_source.lower()
                    })

                    logger.info(
                        f"Saved variation {image.image_name} with batch_id={shared_batch_id}")

                except Exception as e:
                    logger.error(
                        f"Failed to process image {image.id}: {str(e)}")
                    continue

            if not saved_images:
                logger.error("None of the provided images could be saved.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="None of the provided images are valid temporary variations."
                )

            await db.flush()

            # Clean up unsaved variations from the ORIGINAL batches
            if batch_ids:
                unsaved_result = await db.execute(
                    select(Image).where(
                        Image.generation_batch_id.in_(batch_ids),
                        Image.status == ImageStatus.TEMPORARY.value,
                        Image.image_type == Type.VARIATION.value,
                        Image.id.notin_(saved_image_ids)
                    )
                )
                unsaved_variations = unsaved_result.scalars().all()

                if unsaved_variations:
                    logger.info(
                        f"Found {len(unsaved_variations)} unsaved variations to delete")

                    ids_to_delete = [v.id for v in unsaved_variations]

                    logger.info("Breaking FK relationships before deletion...")
                    await db.execute(
                        update(Image)
                        .where(Image.parent_id.in_(ids_to_delete))
                        .values(parent_id=None)
                    )

                    await db.execute(
                        update(Image)
                        .where(Image.root_id.in_(ids_to_delete))
                        .values(root_id=None)
                    )

                    await db.execute(
                        update(Image)
                        .where(Image.lineage_root_id.in_(ids_to_delete))
                        .values(lineage_root_id=None)
                    )

                    await db.flush()

                    for variation in unsaved_variations:
                        try:
                            delete_from_s3(variation.s3_object_key)
                            logger.info(
                                f"Deleted S3 file: {variation.s3_object_key}")
                            delete_from_s3(variation.s3_thumbnail_key)
                            logger.info(
                                f"Deleted S3 thumbnail: {variation.s3_thumbnail_key}")
                        except Exception as e:
                            logger.error(
                                f"Failed to delete S3 files for {variation.id}: {str(e)}")

                    delete_result = await db.execute(
                        delete(Image).where(Image.id.in_(ids_to_delete))
                    )
                    deleted_count = delete_result.rowcount
                    logger.info(
                        f"Deleted {deleted_count} unsaved variations from database")
                else:
                    logger.info("No unsaved variations to delete")
            else:
                logger.warning("No batch_ids found, skipping cleanup")

            await db.commit()
            logger.info(
                f"Successfully saved {len(saved_images)} variations with shared batch_id={shared_batch_id}")

            # Return response with parent image info and validation flag
            response_data = {
                "iteration": next_iteration,
                "iteration_type": "variation",
                "project_id": project_id,
                "prompt": None,
                "enhanced_prompt": None,
                "images": saved_images,
                "variationrootimage": None
            }

            if parent_image:
                response_data["variationrootimage"] = {
                    "id": str(parent_image.id),
                    "display_id": parent_image.image_name,
                    "url": generate_presigned_url(parent_image.s3_object_key),
                    "thumbnail_url": generate_presigned_url(parent_image.s3_thumbnail_key),
                    "is_published": parent_image.is_published,
                    "has_history": True,
                    "source": parent_image.image_source.lower()
                }

            return response_data

        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Exception during saving variations: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save variations."
            )

    async def edit_image(self, db, image, prompt, style_strength, structure_strength, image_type):
        try:
            url = generate_presigned_url(image.s3_object_key)
            logger.info(f"Editing image: {image.image_name}")
            logger.info("Extracting image description...")
            if image_type == "photo":
                description = await extract_description(url=url)
                logger.info("Editing description as per instruction...")
                updated_description = await edit_description(description, prompt)
                logger.info(
                    f"Description updated: {updated_description[:100]}...")
                model = "image4_ultra"
                content_class = "photo"
                negative_prompt = "low quality, worst quality, blurry, out of focus, pixelated, jpeg artifacts, noise, grainy, distorted, compressed, bad anatomy, wrong anatomy, bad proportions, deformed body, malformed limbs, extra limbs, missing limbs, bad hands, deformed hands, extra fingers, missing fingers, fused fingers, twisted fingers, poorly drawn hands, asymmetrical face, misaligned features, deformed face, extra eyes, wrong skin tones, unrealistic textures, elongated body, long neck, disproportionate body, cluttered background, distracting background, random objects, floating objects, watermark, text, NSFW, gore, poor lighting, oversaturated, underexposed, unnatural appearance, mutation"
            elif image_type == "icon":
                description = await extract_icon_description(url=url)
                logger.info("Editing description as per instruction...")
                updated_description = await edit_icon_description(description, prompt)
                logger.info(
                    f"Description updated: {updated_description[:100]}...")
                model = "image4_custom"
                content_class = "art"
                negative_prompt = "multiple lines, broken lines, disconnected strokes, separate line segments, gaps in line, outline style, filled shapes, closed loops, individual elements, multiple strokes, traditional line art, sketchy lines, double lines, overlapping lines, thick outlines, cartoon style, detailed shading, hatching, cross-hatching, filled areas, solid fills, gradient fills, color fills, shadows, highlights, 3D effects, perspective depth, realistic rendering, photographic style, textured surfaces, pattern fills, background elements, decorative details, ornamental features, complex details, intricate patterns, multiple colors, color variations, blurred edges, rough sketches, hand-drawn imperfections, uneven lines, wavy lines, dotted lines, dashed lines, segmented paths, separate shapes, disconnected objects, isolated elements, layered strokes, composite drawings"
            logger.info("Starting image generation with Adobe Firefly API")

            try:
                access_token = await generate_token()
            except Exception as e:
                logger.error(f"Failed to retrieve access token: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve access token."
                )
            if not access_token:
                logger.error("Access token is None or empty")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Access token is invalid."
                )

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": settings.ADOBE_CLIENT_ID,
                "Authorization": f"Bearer {access_token}",
                "x-model-version": model
            }

            payload = {
                "contentClass": content_class,
                "negativePrompt": negative_prompt,
                "numVariations": 1,
                "prompt": updated_description[:1024],
                "size": {"height": image.height, "width": image.width},
                "style": {
                    "imageReference": {"source": {"url": url}},
                    "strength": style_strength
                },
                "structure": {
                    "imageReference": {"source": {"url": url}},
                    "strength": structure_strength
                }
            }

            if image_type == "photo":
                payload["visualIntensity"] = 7
            if image_type == "icon":
                payload["customModelId"] = settings.ADOBE_CUSTOM_MODEL
                payload.pop("style")

            logger.info(
                f"Adobe Firefly payload: {json.dumps(payload, indent=2)}")

            response = requests.post(
                "https://firefly-api.adobe.io/v3/images/generate-async",
                headers=headers,
                data=json.dumps(payload),
                timeout=90
            )
            if response.status_code != 202:
                logger.error(f"Failed to generate image: {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=response.text
                )

            res = response.json()
            status_url = res["statusUrl"]

            logger.info("Polling for image generation status...")
            max_retries = 60
            retry_delay = 2
            image_urls = []

            for attempt in range(max_retries):
                image_fetch_res = get_image(status_url, access_token)
                status_value = image_fetch_res["status"]
                logger.info(
                    f"Attempt {attempt + 1}/{max_retries} - Firefly status: {status_value}")

                if status_value == "succeeded":
                    image_urls = image_fetch_res["result"]["outputs"]
                    break
                elif status_value == "failed":
                    logger.error(f"Image generation failed: {image_fetch_res}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(image_fetch_res)
                    )
                elif status_value in ("pending", "running"):
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Unexpected status: {status_value}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Unexpected status: {status_value}"
                    )
            else:
                logger.error(
                    f"Image generation timed out after {max_retries} retries")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Image generation timed out."
                )

            logger.info("Image generation succeeded, saving to database...")

            if not image_urls or len(image_urls) == 0:
                logger.error("No image returned from Adobe")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No image returned from Adobe"
                )

            img_data = image_urls[0]
            generated_url = img_data.get("image", {}).get("url")
            seed_value = img_data.get("seed")

            if not generated_url:
                logger.error("No URL in image data")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No image URL in response"
                )

            # Download image
            logger.info(
                f"Downloading generated image from: {generated_url[:100]}...")
            resp = await asyncio.to_thread(requests.get, generated_url, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Failed to download image: {resp.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to download generated image"
                )

            image_bytes = resp.content
            logger.info(f"Downloaded image size: {len(image_bytes)} bytes")

            # Generate S3 keys
            uid = uuid.uuid4().hex[:8]
            file_key = f"gilead/imagegen/editedimages/originals/image_{uid}.jpg"
            thumb_key = f"gilead/imagegen/editedimages/thumbnails/thumb_{uid}.webp"

            # Upload original to S3
            logger.info(f"Uploading to S3: {file_key}")
            upload_to_s3(io.BytesIO(image_bytes), file_key,
                         content_type="image/jpeg")

            # Generate and upload thumbnail
            logger.info("Generating thumbnail...")
            img = PILImage.open(io.BytesIO(image_bytes))
            img.thumbnail(THUMBNAIL_MAX_SIZE)
            thumb_buffer = io.BytesIO()
            img.save(thumb_buffer, format="WEBP", optimize=True, quality=100)
            thumb_buffer.seek(0)

            logger.info(f"Uploading thumbnail: {thumb_key}")
            upload_to_s3(thumb_buffer, thumb_key, content_type="image/webp")

            # CHANGED: Get lineage context
            lineage_context = get_root_and_parent(image)
            logger.info(f"Creating edited image with lineage: root_id={lineage_context['root_id']}, "
                        f"parent_id={lineage_context['parent_id']}, "
                        f"lineage_root_id={lineage_context['lineage_root_id']}, "
                        f"version_source_id={image.id}")

            image_name = await generate_next_image_name(db)

            # CHANGED: Create Image with lineage fields
            image_obj = Image(
                user_id=image.user_id,
                width=image.width,
                height=image.height,
                image_name=image_name,
                s3_object_key=file_key,
                s3_thumbnail_key=thumb_key,
                prompt_id=image.prompt_id,
                project_id=image.project_id,
                style_reference_id=image.style_reference_id,
                composition_reference_id=image.composition_reference_id,
                seed=seed_value,
                status=ImageStatus.TEMPORARY.value,
                image_type=Type.EDITED.value,
                image_source=ImageSource.LOCAL.value,
                # Lineage tracking
                root_id=lineage_context['root_id'],
                parent_id=lineage_context['parent_id'],
                lineage_root_id=lineage_context['lineage_root_id'],  # NEW

                version_source_id=image.id,  # Created FROM this image
                # CHANGED: Inherit from parent (not None)
                version_number=image.version_number,
                lineage_version_number=None,  # NEW: Will be assigned when saved

                is_base=False,  # Not a base version yet
                generation_batch_id=uuid.uuid4(),
            )

            db.add(image_obj)
            await db.flush()
            logger.info(
                f"Created database record: {image_obj.image_name} (ID: {image_obj.id})")

            # Generate presigned URLs for response
            response_url = generate_presigned_url(file_key)
            thumb_response_url = generate_presigned_url(thumb_key)

            # Commit transaction
            await db.commit()

            logger.info(
                f"Successfully saved temporary edit: {image_obj.image_name}")

            return {
                "id": str(image_obj.id),
                "display_id": image_obj.image_name,
                "url": response_url,
                "thumbnail_url": thumb_response_url,
                "is_published": image_obj.is_published,
                "has_history": False,
                "source": image_obj.image_source.lower()
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Exception during image editing: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Image editing failed."
            )

    async def save_edited_image(self, db, id):
        try:
            result = await db.execute(select(Image).filter(Image.id == id))
            image = result.scalar_one_or_none()

            if image.status != ImageStatus.TEMPORARY.value:
                logger.warning(
                    f"Image is not temporary (status: {image.status})")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Image is not a temporary edit."
                )

            if image.image_type != Type.EDITED.value:
                logger.warning(
                    f"Image is not an edited image (type: {image.image_type})")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Image is not an edited image."
                )

            root_result = await db.execute(
                select(Image).where(Image.id == image.lineage_root_id)
            )
            lineage_root = root_result.scalar_one_or_none()

            if not lineage_root:
                logger.error(f"Lineage root {image.lineage_root_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Lineage root not found"
                )

            inherited_iteration = lineage_root.iteration_number

            logger.info(
                f"[Lineage {lineage_root.image_name}] Inheriting iteration_number: {inherited_iteration} for edit {image.image_name}")

            # Check if lineage root has prompt and validate preset_dict
            validate = False
            lineage_root_prompt_id = lineage_root.prompt_id

            if lineage_root_prompt_id:
                prompt_record = await db.execute(
                    select(Prompt).where(Prompt.id == lineage_root_prompt_id)
                )
                prompt_result = prompt_record.scalar_one_or_none()

                if prompt_result and prompt_result.guidelines_applied:
                    validate = True
                    logger.info(
                        f"Lineage root prompt has preset_dict - validation enabled")
                else:
                    validate = False
                    logger.info(
                        f"Lineage root prompt has no preset_dict - validation disabled")
            else:
                logger.info(f"Lineage root has no prompt_id")

            logger.info(
                f"Calculating lineage version for lineage_root_id: {image.lineage_root_id}")

            result = await db.execute(
                select(func.max(Image.lineage_version_number)).where(
                    or_(
                        Image.id == image.lineage_root_id,
                        Image.lineage_root_id == image.lineage_root_id
                    ),
                    Image.status == ImageStatus.PERMANENT.value
                )
            )
            max_lineage_version = result.scalar()

            if max_lineage_version is None:
                logger.warning(
                    f"No versions found for lineage {image.lineage_root_id}, defaulting to 0")
                max_lineage_version = 0

            new_lineage_version = max_lineage_version + 1
            logger.info(
                f"Assigning lineage version {new_lineage_version} to {image.image_name} (max was {max_lineage_version})")

            saved_image_id = image.id
            lineage_root_id = image.lineage_root_id

            # Mark previous EDITS as is_base=False (not originals/variations)
            logger.info(
                f"Marking previous edited versions in lineage as is_base=False")
            await db.execute(
                update(Image)
                .where(
                    Image.lineage_root_id == lineage_root_id,
                    Image.image_type == Type.EDITED.value,
                    Image.is_base == True
                )
                .values(is_base=False)
            )

            await db.execute(
                update(Image)
                .where(
                    Image.id == lineage_root_id,
                    Image.is_base == True
                )
                .values(is_base=False)
            )
            logger.info(
                f"Marked lineage root {lineage_root.image_name} as is_base=False (superseded by edit)")

            # Associate edited image with lineage root's prompt_id
            if lineage_root_prompt_id:
                image.prompt_id = lineage_root_prompt_id
                logger.info(
                    f"Associated {image.image_name} with lineage root prompt_id: {lineage_root_prompt_id}")

            image.status = ImageStatus.PERMANENT.value
            image.lineage_version_number = new_lineage_version
            image.is_base = True
            image.iteration_number = inherited_iteration

            await db.flush()

            logger.info(f"Saved edit: ID={saved_image_id}, Name={image.image_name}, "
                        f"Version={new_lineage_version}, iteration={inherited_iteration} (inherited), is_base=True")

            unsaved_result = await db.execute(
                select(Image).where(
                    Image.lineage_root_id == lineage_root_id,
                    Image.status == ImageStatus.TEMPORARY.value,
                    Image.image_type == Type.EDITED.value,
                    Image.id != saved_image_id
                )
            )
            unsaved_edits = unsaved_result.scalars().all()

            logger.info(
                f"Found {len(unsaved_edits)} temporary edits to delete")

            if unsaved_edits:
                ids_to_delete = [unsaved.id for unsaved in unsaved_edits]

                # Break FK relationships to prevent cascade deletion
                logger.info(
                    "Breaking FK relationships to prevent cascade deletion...")

                await db.execute(
                    update(Image)
                    .where(Image.parent_id.in_(ids_to_delete))
                    .values(parent_id=None)
                )

                await db.execute(
                    update(Image)
                    .where(Image.root_id.in_(ids_to_delete))
                    .values(root_id=None)
                )

                await db.execute(
                    update(Image)
                    .where(Image.lineage_root_id.in_(ids_to_delete))
                    .values(lineage_root_id=None)
                )

                await db.flush()
                logger.info("FK relationships broken successfully")

                # Delete from S3
                for unsaved in unsaved_edits:
                    try:
                        delete_from_s3(unsaved.s3_object_key)
                        delete_from_s3(unsaved.s3_thumbnail_key)
                        logger.info(
                            f"Deleted S3 files for {unsaved.image_name}")
                    except Exception as e:
                        logger.error(f"Failed to delete S3 files: {str(e)}")

                # Delete from database
                delete_result = await db.execute(
                    delete(Image).where(Image.id.in_(ids_to_delete))
                )
                deleted_count = delete_result.rowcount
                logger.info(
                    f"Deleted {deleted_count} unsaved edits from database")

            await db.commit()

            # Generate presigned URLs
            url = generate_presigned_url(image.s3_object_key)
            thumbnail_url = generate_presigned_url(image.s3_thumbnail_key)

            logger.info(
                f"Successfully saved {image.image_name} as lineage version {new_lineage_version}, iteration {inherited_iteration} (inherited)")

            return {
                "id": str(image.id),
                "display_id": image.image_name,
                "url": url,
                "thumbnail_url": thumbnail_url,
                "is_published": image.is_published,
                "source": image.image_source.lower(),
                "has_history": True,
                "can_validate": validate,
                "iteration_number": inherited_iteration,
                "lineage_version_number": new_lineage_version
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Exception saving edited image: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save edited image."
            )

    async def get_image_family(self, db, image_id: str):
        try:
            logger.info(f"Fetching image family for ID: {image_id}")

            result = await db.execute(
                select(Image).where(Image.id == image_id)
            )
            selected_image = result.scalar_one_or_none()

            if not selected_image:
                logger.error(f"Image with id {image_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Image not found"
                )

            logger.info(f"Selected image: {selected_image.image_name}, "
                        f"type: {selected_image.image_type}, "
                        f"lineage_root_id: {selected_image.lineage_root_id}")

            # Determine which lineage tree this image belongs to
            if selected_image.image_type == Type.VARIATION.value:
                lineage_root_id = selected_image.id
                include_variation_as_root = True
                selected_version_number = selected_image.lineage_version_number or 1
                logger.info(
                    f"Variation detected - lineage root: {lineage_root_id}")

            elif selected_image.image_type in [Type.ORIGINAL.value, Type.LIBRARY.value]:
                lineage_root_id = selected_image.id
                include_variation_as_root = False
                selected_version_number = selected_image.lineage_version_number or 1
                logger.info(
                    f"Original/Library detected - lineage root: {lineage_root_id}")

            elif selected_image.image_type == Type.EDITED.value:
                lineage_root_id = selected_image.lineage_root_id
                selected_version_number = selected_image.lineage_version_number
                logger.info(f"Edit detected - lineage root: {lineage_root_id}")

                # Check if we're editing a variation or an original/library image
                root_result = await db.execute(
                    select(Image).where(Image.id == lineage_root_id)
                )
                lineage_root_image = root_result.scalar_one_or_none()

                include_variation_as_root = (
                    lineage_root_image and
                    lineage_root_image.image_type == Type.VARIATION.value
                )
                logger.info(
                    f"Root is {'variation' if include_variation_as_root else 'original/library'}")

            else:
                logger.error(
                    f"Unknown image type: {selected_image.image_type}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unknown image type"
                )

            # Fetch all versions in this lineage tree
            if include_variation_as_root:
                # Variation tree: includes only the variation and its edits
                base_versions_result = await db.execute(
                    select(Image).where(
                        or_(
                            Image.id == lineage_root_id,
                            and_(
                                Image.lineage_root_id == lineage_root_id,
                                Image.image_type == Type.EDITED.value
                            )
                        ),
                        Image.status == ImageStatus.PERMANENT.value
                    ).order_by(Image.lineage_version_number.desc())
                )
            else:
                # Original/Library tree: includes root and all edits
                base_versions_result = await db.execute(
                    select(Image).where(
                        or_(
                            Image.id == lineage_root_id,
                            Image.lineage_root_id == lineage_root_id
                        ),
                        Image.status == ImageStatus.PERMANENT.value,
                        Image.image_type.in_([
                            Type.ORIGINAL.value,
                            Type.LIBRARY.value,
                            Type.EDITED.value
                        ])
                    ).order_by(Image.lineage_version_number.desc())
                )

            base_versions = base_versions_result.scalars().all()

            if not base_versions:
                logger.warning(
                    f"No base versions found for lineage_root_id: {lineage_root_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No versions found for this image family"
                )
            logger.info(f"Found {len(base_versions)} versions in lineage")
            max_version = max(
                [v.lineage_version_number for v in base_versions])
            base_version_ids = [v.id for v in base_versions]
            # Batch fetch all variations to avoid N+1 queries
            all_variations_result = await db.execute(
                select(Image).where(
                    Image.parent_id.in_(base_version_ids),
                    Image.image_type == Type.VARIATION.value,
                    Image.status == ImageStatus.PERMANENT.value
                ).order_by(Image.parent_id, Image.created_at)
            )
            all_variations = all_variations_result.scalars().all()

            variations_by_parent = {}
            for variation in all_variations:
                if variation.parent_id not in variations_by_parent:
                    variations_by_parent[variation.parent_id] = []
                variations_by_parent[variation.parent_id].append(variation)

            logger.info(f"Fetched {len(all_variations)} variations in batch")

            # Batch fetch style references
            style_ref_ids = [
                v.style_reference_id for v in base_versions if v.style_reference_id]
            style_refs_map = {}

            if style_ref_ids:
                try:
                    style_refs_result = await db.execute(
                        select(StyleReference).where(
                            StyleReference.id.in_(style_ref_ids))
                    )
                    style_refs = style_refs_result.scalars().all()

                    for ref in style_refs:
                        try:
                            style_refs_map[ref.id] = generate_presigned_url(
                                ref.s3_object_key)
                        except ClientError as e:
                            logger.error(
                                f"S3 error for style ref {ref.id}: {e}")
                            style_refs_map[ref.id] = None

                    logger.info(
                        f"Batch fetched {len(style_refs_map)} style references")
                except Exception as e:
                    logger.error(f"Error batch fetching style references: {e}")

            # Batch fetch composition references
            comp_ref_ids = [
                v.composition_reference_id for v in base_versions if v.composition_reference_id]
            comp_refs_map = {}
            if comp_ref_ids:
                try:
                    comp_refs_result = await db.execute(
                        select(CompositionReference).where(
                            CompositionReference.id.in_(comp_ref_ids))
                    )
                    comp_refs = comp_refs_result.scalars().all()

                    for ref in comp_refs:
                        try:
                            comp_refs_map[ref.id] = generate_presigned_url(
                                ref.s3_object_key)
                        except ClientError as e:
                            logger.error(
                                f"S3 error for composition ref {ref.id}: {e}")
                            comp_refs_map[ref.id] = None

                    logger.info(
                        f"Batch fetched {len(comp_refs_map)} composition references")
                except Exception as e:
                    logger.error(
                        f"Error batch fetching composition references: {e}")
            # Build the version history response
            edit_history = []
            for base_version in base_versions:
                variations = variations_by_parent.get(base_version.id, [])
                is_current = (
                    base_version.lineage_version_number == max_version)
                style_url = style_refs_map.get(
                    base_version.style_reference_id) if base_version.style_reference_id else None
                comp_url = comp_refs_map.get(
                    base_version.composition_reference_id) if base_version.composition_reference_id else None
                version_obj = {
                    "id": str(base_version.id),
                    "version": format_version(base_version.lineage_version_number),
                    "display_id": base_version.image_name,
                    "is_current": is_current,
                    "url": generate_presigned_url(base_version.s3_object_key),
                    "thumbnail_url": generate_presigned_url(base_version.s3_thumbnail_key),
                    "is_published": base_version.is_published,
                    "source": base_version.image_source.lower(),
                    "created_at": format_date(base_version.created_at.isoformat()),
                    "style_reference_url": style_url,
                    "composition_reference_url": comp_url,
                    "variations": [
                        {
                            "id": str(v.id),
                            "display_id": v.image_name,
                            "url": generate_presigned_url(v.s3_object_key),
                            "thumbnail_url": generate_presigned_url(v.s3_thumbnail_key),
                            "is_published": v.is_published,
                            "source": v.image_source.lower()
                        }
                        for v in variations
                    ]
                }
                edit_history.append(version_obj)
            logger.info(
                f"Successfully built family tree with {len(edit_history)} versions")
            return edit_history
        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Exception in get_image_family: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch image family."
            )

    async def list_all_projects(
        self,
        db: AsyncSession,
        product: Optional[str] = None,
        country: Optional[str] = None,
        type: Optional[str] = None,
        search: Optional[str] = None,
        skip: int = 0,
        limit: int = 10
    ) -> Dict[str, Any]:
        try:
            filters = [Project.is_active == True]
            if product:
                filters.append(Project.product == product)
            if country:
                filters.append(Project.country == country)
            if type:
                filters.append(Project.type == type)
            if search:
                filters.append(Project.name.ilike(f"%{search}%"))
            count_query = select(func.count(Project.id)).where(*filters)
            count_result = await db.execute(count_query)
            total_count = count_result.scalar() or 0

            if total_count == 0:
                return {"projects": [], "total": 0}
            project_query = (
                select(Project)
                .where(*filters)
                .order_by(desc(Project.created_at))
                .offset(skip)
                .limit(limit)
            )
            result = await db.execute(project_query)
            projects = result.scalars().all()

            if not projects:
                return {"projects": [], "total": total_count}

            project_ids = [p.id for p in projects]

            representative_images_query = select(
                Image.project_id,
                Image.image_name,
                Image.s3_thumbnail_key,
                Image.prompt_id,
                Image.is_published,
                func.row_number().over(
                    partition_by=Image.project_id,
                    order_by=[
                        desc(Image.is_published),  # Published first
                        asc(Image.created_at),     # Then oldest
                        asc(Image.image_name)
                    ]
                ).label('row_num')
            ).where(
                Image.project_id.in_(project_ids),
                Image.status == ImageStatus.PERMANENT.value
            ).subquery()
            rep_query = (
                select(
                    representative_images_query.c.project_id,
                    representative_images_query.c.image_name,
                    representative_images_query.c.s3_thumbnail_key,
                    representative_images_query.c.prompt_id
                )
                .where(representative_images_query.c.row_num == 1)
            )
            rep_result = await db.execute(rep_query)
            rep_images = {row.project_id: row for row in rep_result}
            total_images_query = (
                select(
                    Image.project_id,
                    func.count(Image.id).label('total_count')
                )
                .where(
                    Image.project_id.in_(project_ids),
                    Image.status == ImageStatus.PERMANENT.value
                )
                .group_by(Image.project_id)
            )
            total_result = await db.execute(total_images_query)
            total_images_map = {
                row.project_id: row.total_count for row in total_result}
            published_images_query = (
                select(
                    Image.project_id,
                    func.count(Image.id).label('published_count')
                )
                .where(
                    Image.project_id.in_(project_ids),
                    Image.is_published == True,
                    Image.status == ImageStatus.PERMANENT.value
                )
                .group_by(Image.project_id)
            )
            published_result = await db.execute(published_images_query)
            published_images_map = {
                row.project_id: row.published_count for row in published_result}
            prompt_ids = [
                img.prompt_id for img in rep_images.values() if img.prompt_id]
            prompts_map = {}
            if prompt_ids:
                prompts_query = select(Prompt).where(Prompt.id.in_(prompt_ids))
                prompts_result = await db.execute(prompts_query)
                prompts_map = {
                    p.id: p.prompt for p in prompts_result.scalars()}
            projects_data = []
            for project in projects:
                rep_image = rep_images.get(project.id)
                projects_data.append({
                    "project_id": str(project.id),
                    "project_name": f"{project.name}_{project.code:05d}" if project.code is not None else None,
                    "product": project.product,
                    "country": project.country,
                    "total_images": total_images_map.get(project.id, 0),
                    "published_images": published_images_map.get(project.id, 0),
                    "url": generate_presigned_url(rep_image.s3_thumbnail_key) if rep_image else None,
                    "prompt": prompts_map.get(rep_image.prompt_id) if rep_image and rep_image.prompt_id else None,
                    "image_name": rep_image.image_name if rep_image else None
                })
            logger.info(
                f"Fetched {len(projects_data)} projects (total: {total_count})")
            return {
                "projects": projects_data,
                "total": total_count
            }
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Exception in list_all_projects: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch projects."
            )

    async def get_project_history(self, db: AsyncSession, project_id: str, skip: int = 0, limit: int = 10):
        try:
            logger.info(
                f"Fetching history for project: {project_id}, skip: {skip}, limit: {limit}")

            # STEP 1: VALIDATE PROJECT
            project_result = await db.execute(
                select(Project).where(
                    Project.id == project_id,
                    Project.is_active == True
                )
            )
            project = project_result.scalar_one_or_none()

            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )

            # STEP 2: COUNT TOTAL ITERATIONS
            total_iterations_result = await db.execute(
                select(func.count(func.distinct(Image.iteration_number)))
                .where(
                    Image.project_id == project_id,
                    Image.status == ImageStatus.PERMANENT.value,
                    Image.iteration_number.isnot(None)
                )
            )
            total_iterations = total_iterations_result.scalar()

            logger.info(f"Total iterations: {total_iterations}")

            # STEP 3: FETCH PAGINATED ITERATION NUMBERS
            iterations_result = await db.execute(
                select(
                    Image.iteration_number,
                    func.min(Image.created_at).label('created_at')
                )
                .where(
                    Image.project_id == project_id,
                    Image.status == ImageStatus.PERMANENT.value,
                    Image.iteration_number.isnot(None)
                )
                .group_by(Image.iteration_number)
                .order_by(Image.iteration_number.asc())
                .offset(skip)
                .limit(limit)
            )
            iterations = iterations_result.all()

            if not iterations:
                logger.info(
                    f"No iterations found for project {project_id} with skip={skip}, limit={limit}")
                return {
                    "data": [],
                    "total": total_iterations,
                    "returned": 0
                }

            logger.info(f"Found {len(iterations)} iterations")

            iteration_numbers = [it.iteration_number for it in iterations]

            # STEP 4: BATCH FETCH ALL IMAGES (only latest versions via is_base=True)
            all_images_result = await db.execute(
                select(Image).where(
                    Image.project_id == project_id,
                    Image.iteration_number.in_(iteration_numbers),
                    Image.status == ImageStatus.PERMANENT.value,
                    Image.is_base == True
                ).order_by(Image.iteration_number, Image.created_at)
            )
            all_images = all_images_result.scalars().all()

            # STEP 5: GROUP IMAGES BY ITERATION
            images_by_iteration = {}
            for img in all_images:
                if img.iteration_number not in images_by_iteration:
                    images_by_iteration[img.iteration_number] = []
                images_by_iteration[img.iteration_number].append(img)

            logger.info(
                f"Batch fetched {len(all_images)} images across {len(iterations)} iterations")

            # STEP 6: BATCH FETCH PROMPTS FOR VALIDATION
            prompt_ids = list(
                set([img.prompt_id for img in all_images if img.prompt_id]))
            prompt_validation_map = {}

            if prompt_ids:
                try:
                    prompts_result = await db.execute(
                        select(Prompt).where(Prompt.id.in_(prompt_ids))
                    )
                    prompts = prompts_result.scalars().all()

                    for prompt in prompts:
                        prompt_validation_map[prompt.id] = bool(
                            prompt.guidelines_applied)

                    logger.info(
                        f"Batch fetched {len(prompt_validation_map)} prompts for validation")
                except Exception as e:
                    logger.error(f"Error batch fetching prompts: {e}")

            # STEP 7: BATCH FETCH VARIATION COUNTS
            all_image_ids = [img.id for img in all_images]

            variations_count_result = await db.execute(
                select(
                    Image.parent_id,
                    func.count(Image.id).label('variation_count')
                )
                .where(
                    Image.parent_id.in_(all_image_ids),
                    Image.image_type == Type.VARIATION.value,
                    Image.status == ImageStatus.PERMANENT.value
                )
                .group_by(Image.parent_id)
            )
            variations_count_map = {
                row.parent_id: row.variation_count for row in variations_count_result.all()}

            # STEP 8: BATCH FETCH LINEAGE INFO
            lineage_roots = list(
                set([img.lineage_root_id for img in all_images if img.lineage_root_id]))
            lineage_version_counts = {}

            if lineage_roots:
                lineage_counts_result = await db.execute(
                    select(
                        Image.lineage_root_id,
                        func.max(Image.lineage_version_number).label(
                            'max_version')
                    )
                    .where(
                        Image.lineage_root_id.in_(lineage_roots),
                        Image.status == ImageStatus.PERMANENT.value
                    )
                    .group_by(Image.lineage_root_id)
                )
                lineage_version_counts = {
                    row.lineage_root_id: row.max_version for row in lineage_counts_result.all()}

            logger.info(f"Batch fetched variation counts and lineage info")

            # STEP 9: BATCH FETCH LINEAGE ROOT TYPES (for EDITED images)
            lineage_root_types = {}
            if lineage_roots:
                root_types_result = await db.execute(
                    select(Image.id, Image.image_type).where(
                        Image.id.in_(lineage_roots)
                    )
                )
                lineage_root_types = {
                    row.id: row.image_type for row in root_types_result.all()}
                logger.info(f"Batch fetched {len(lineage_root_types)} lineage root types")

            # Helper function to determine has_history
            def check_has_history(img):
                has_variations = variations_count_map.get(img.id, 0) > 0
                has_ancestors = (img.lineage_version_number or 1) > 1
                max_version_in_lineage = lineage_version_counts.get(
                    img.lineage_root_id, img.lineage_version_number or 1)
                has_successors = (
                    img.lineage_version_number or 1) < max_version_in_lineage

                return has_variations or has_ancestors or has_successors

            # Helper function to check if image can be validated
            def check_can_validate(img):
                if img.prompt_id:
                    return prompt_validation_map.get(img.prompt_id, False)
                return False

            history = []

            # STEP 10: BUILD RESPONSE FOR EACH ITERATION
            for iteration in iterations:
                iteration_num = iteration.iteration_number
                images = images_by_iteration.get(iteration_num, [])

                if not images:
                    continue

                first_image = images[0]
                image_type = first_image.image_type

                # Determine the base iteration type (handle EDITED images)
                iteration_base_type = image_type
                
                if image_type == Type.EDITED.value:
                    # For EDITED images, check the lineage root's original type
                    root_id = first_image.lineage_root_id
                    if root_id and root_id in lineage_root_types:
                        iteration_base_type = lineage_root_types[root_id]
                        logger.debug(f"Iteration {iteration_num}: EDITED image, root type={iteration_base_type}")
                    else:
                        # Fallback: treat as ORIGINAL
                        iteration_base_type = Type.ORIGINAL.value
                        logger.debug(f"Iteration {iteration_num}: EDITED image with no root, defaulting to ORIGINAL")

                logger.debug(f"Processing iteration {iteration_num} with base type {iteration_base_type}")

                # HANDLE VARIATION ITERATIONS
                if iteration_base_type == Type.VARIATION.value:
                    logger.debug(
                        f"Processing VARIATION iteration {iteration_num}")

                    first_variation = images[0]
                    parent = None
                    parent_prompt_text = None
                    parent_enhanced_prompt = None

                    if first_variation.parent_id:
                        parent_result = await db.execute(
                            select(Image).where(
                                or_(
                                    Image.id == first_variation.parent_id,
                                    Image.lineage_root_id == first_variation.parent_id
                                ),
                                Image.status == ImageStatus.PERMANENT.value,
                                Image.is_base == True
                            ).order_by(Image.lineage_version_number.desc())
                            .limit(1)
                        )
                        parent = parent_result.scalar_one_or_none()

                        if parent and parent.prompt_id:
                            prompt_result = await db.execute(
                                select(Prompt).where(
                                    Prompt.id == parent.prompt_id)
                            )
                            parent_prompt_obj = prompt_result.scalar_one_or_none()
                            if parent_prompt_obj:
                                parent_prompt_text = parent_prompt_obj.prompt
                                parent_enhanced_prompt = parent_prompt_obj.enhanced_prompt

                    variations_data = []
                    for var in images:
                        variations_data.append({
                            "id": str(var.id),
                            "display_id": var.image_name,
                            "url": generate_presigned_url(var.s3_object_key),
                            "thumbnail_url": generate_presigned_url(var.s3_thumbnail_key),
                            "is_published": var.is_published,
                            "has_history": check_has_history(var),
                            "can_validate": check_can_validate(var)
                        })

                    variationrootimage = None
                    if parent:
                        variationrootimage = {
                            "id": str(parent.id),
                            "display_id": parent.image_name,
                            "url": generate_presigned_url(parent.s3_object_key),
                            "thumbnail_url": generate_presigned_url(parent.s3_thumbnail_key),
                            "style_reference_url": await get_style_reference_url(db, parent.style_reference_id)
                            if parent.style_reference_id else None,
                            "composition_reference_url": await get_composition_reference_url(db, parent.composition_reference_id)
                            if parent.composition_reference_id else None,
                            "is_published": parent.is_published,
                            "source": parent.image_source.lower(),
                            "has_history": check_has_history(parent),
                            "can_validate": check_can_validate(parent)
                        }

                    history.append({
                        "iteration": iteration_num,
                        "iteration_type": "variation",
                        "project_id": project_id,
                        "prompt": parent_prompt_text,
                        "enhanced_prompt": parent_enhanced_prompt,
                        "images": variations_data,
                        "variationrootimage": variationrootimage
                    })

                # HANDLE LIBRARY ITERATIONS
                elif iteration_base_type == Type.LIBRARY.value:
                    logger.debug(
                        f"Processing LIBRARY iteration {iteration_num}")

                    prompt = None
                    enhanced_prompt = None

                    if first_image.prompt_id:
                        prompt_result = await db.execute(
                            select(Prompt).where(
                                Prompt.id == first_image.prompt_id)
                        )
                        prompt_obj = prompt_result.scalar_one_or_none()
                        if prompt_obj:
                            prompt = prompt_obj.prompt
                            enhanced_prompt = prompt_obj.enhanced_prompt
                            logger.debug(
                                f"Library image has associated prompt")
                    else:
                        logger.debug(f"Library image has no prompt")

                    images_data = []
                    for img in images:
                        images_data.append({
                            "id": str(img.id),
                            "display_id": img.image_name,
                            "url": generate_presigned_url(img.s3_object_key),
                            "thumbnail_url": generate_presigned_url(img.s3_thumbnail_key),
                            "style_reference_url": await get_style_reference_url(db, img.style_reference_id)
                            if img.style_reference_id else None,
                            "composition_reference_url": await get_composition_reference_url(db, img.composition_reference_id)
                            if img.composition_reference_id else None,
                            "is_published": img.is_published,
                            "source": img.image_source.lower(),
                            "has_history": check_has_history(img),
                            "can_validate": check_can_validate(img)
                        })

                    history.append({
                        "iteration": iteration_num,
                        "iteration_type": "library",
                        "project_id": project_id,
                        "prompt": prompt,
                        "enhanced_prompt": enhanced_prompt,
                        "images": images_data,
                        "variationrootimage": None
                    })

                # HANDLE ORIGINAL/GENERATION ITERATIONS (includes EDITED from ORIGINAL)
                else:
                    logger.debug(
                        f"Processing ORIGINAL/GENERATION iteration {iteration_num}")

                    prompt = None
                    enhanced_prompt = None

                    if first_image.prompt_id:
                        prompt_result = await db.execute(
                            select(Prompt).where(
                                Prompt.id == first_image.prompt_id)
                        )
                        prompt_obj = prompt_result.scalar_one_or_none()
                        if prompt_obj:
                            prompt = prompt_obj.prompt
                            enhanced_prompt = prompt_obj.enhanced_prompt

                    images_data = []
                    for img in images:
                        images_data.append({
                            "id": str(img.id),
                            "display_id": img.image_name,
                            "url": generate_presigned_url(img.s3_object_key),
                            "thumbnail_url": generate_presigned_url(img.s3_thumbnail_key),
                            "style_reference_url": await get_style_reference_url(db, img.style_reference_id)
                            if img.style_reference_id else None,
                            "composition_reference_url": await get_composition_reference_url(db, img.composition_reference_id)
                            if img.composition_reference_id else None,
                            "is_published": img.is_published,
                            "source": img.image_source.lower(),
                            "has_history": check_has_history(img),
                            "can_validate": check_can_validate(img)
                        })

                    history.append({
                        "iteration": iteration_num,
                        "iteration_type": "generation",
                        "project_id": project_id,
                        "prompt": prompt,
                        "enhanced_prompt": enhanced_prompt,
                        "images": images_data,
                        "variationrootimage": None
                    })

            logger.info(
                f"Successfully built history with {len(history)} iterations")
            return {
                "data": history,
                "total": total_iterations,
                "returned": len(history)
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Exception in get_project_history: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch project history"
            )


    async def get_countries(self, db):
        try:
            result = await db.execute(select(GeographicEntities).order_by(GeographicEntities.key.asc()))
            countries = result.scalars().all()
            data = [
                {"key": country.key, "value": country.value}
                for country in countries
            ]
            return data
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"An Exception Occurred in Countries API: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=settings.err_msg
            )

    async def generate_compliant_image(self, db, image, prompt_obj, prompt):
        try:
            enhanced_prompt = await restructure_prompt(prompt=prompt)
            try:
                access_token = await generate_token()
            except Exception as e:
                logger.error(
                    f"An exception occurred while retrieving the access token: {str(e)}"
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve access token.",
                )
            if not access_token:
                logger.error("Access token is None or empty.")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Access token is invalid.",
                )
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": settings.ADOBE_CLIENT_ID,
                "Authorization": f"Bearer {access_token}",
                "x-model-version": "image4_ultra"
            }

            payload = {
                "contentClass": "photo",
                "negativePrompt": "low quality, worst quality, blurry, out of focus, pixelated, jpeg artifacts, noise, grainy, distorted, compressed, bad anatomy, wrong anatomy, bad proportions, deformed body, malformed limbs, extra limbs, missing limbs, bad hands, deformed hands, extra fingers, missing fingers, fused fingers, twisted fingers, poorly drawn hands, asymmetrical face, misaligned features, deformed face, extra eyes, wrong skin tones, unrealistic textures, elongated body, long neck, disproportionate body, cluttered background, distracting background, random objects, floating objects, watermark, text, NSFW, gore, poor lighting, oversaturated, underexposed, unnatural appearance, mutation",
                "numVariations": 1,
                "prompt": enhanced_prompt[:1024],
                "size": {"height": image.height, "width": image.width},
                "visualIntensity": 7,
                "seeds": [
                    image.seed
                ]
            }

            logger.info(
                f"Adobe Firefly payload: {json.dumps(payload, indent=2)}")

            response = requests.post(
                "https://firefly-api.adobe.io/v3/images/generate-async",
                headers=headers,
                data=json.dumps(payload),
                timeout=90
            )
            if response.status_code != 202:
                logger.error(f"Failed to generate image: {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=response.text
                )

            res = response.json()
            status_url = res["statusUrl"]

            logger.info("Polling for image generation status...")
            max_retries = 60
            retry_delay = 2
            image_urls = []

            for attempt in range(max_retries):
                image_fetch_res = get_image(status_url, access_token)
                status_value = image_fetch_res["status"]
                logger.info(
                    f"Attempt {attempt + 1}/{max_retries} - Firefly status: {status_value}")

                if status_value == "succeeded":
                    image_urls = image_fetch_res["result"]["outputs"]
                    break
                elif status_value == "failed":
                    logger.error(f"Image generation failed: {image_fetch_res}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(image_fetch_res)
                    )
                elif status_value in ("pending", "running"):
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Unexpected status: {status_value}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Unexpected status: {status_value}"
                    )
            else:
                logger.error(
                    f"Image generation timed out after {max_retries} retries")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Image generation timed out."
                )
            logger.info("Image generation succeeded, saving to database...")
            if not image_urls or len(image_urls) == 0:
                logger.error("No image returned from Adobe")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No image returned from Adobe"
                )

            img_data = image_urls[0]
            generated_url = img_data.get("image", {}).get("url")
            seed_value = img_data.get("seed")

            if not generated_url:
                logger.error("No URL in image data")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No image URL in response"
                )

            # Download image
            logger.info(
                f"Downloading generated image from: {generated_url[:100]}...")
            resp = await asyncio.to_thread(requests.get, generated_url, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Failed to download image: {resp.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to download generated image"
                )

            image_bytes = resp.content
            logger.info(f"Downloaded image size: {len(image_bytes)} bytes")

            # Generate S3 keys
            uid = uuid.uuid4().hex[:8]
            file_key = f"gilead/imagegen/editedimages/originals/image_{uid}.jpg"
            thumb_key = f"gilead/imagegen/editedimages/thumbnails/thumb_{uid}.webp"

            # Upload original to S3
            logger.info(f"Uploading to S3: {file_key}")
            upload_to_s3(io.BytesIO(image_bytes), file_key,
                         content_type="image/jpeg")

            # Generate and upload thumbnail
            logger.info("Generating thumbnail...")
            img = PILImage.open(io.BytesIO(image_bytes))
            img.thumbnail(THUMBNAIL_MAX_SIZE)
            thumb_buffer = io.BytesIO()
            img.save(thumb_buffer, format="WEBP", optimize=True, quality=100)
            thumb_buffer.seek(0)

            logger.info(f"Uploading thumbnail: {thumb_key}")
            upload_to_s3(thumb_buffer, thumb_key, content_type="image/webp")

            # CHANGED: Get lineage context
            lineage_context = get_root_and_parent(image)
            logger.info(f"Creating edited image with lineage: root_id={lineage_context['root_id']}, "
                        f"parent_id={lineage_context['parent_id']}, "
                        f"lineage_root_id={lineage_context['lineage_root_id']}, "
                        f"version_source_id={image.id}")

            image_name = await generate_next_image_name(db)
            # CHANGED: Create Image with lineage fields
            image_obj = Image(
                user_id=image.user_id,
                width=image.width,
                height=image.height,
                image_name=image_name,
                s3_object_key=file_key,
                s3_thumbnail_key=thumb_key,
                prompt_id=image.prompt_id,
                project_id=image.project_id,
                style_reference_id=image.style_reference_id,
                composition_reference_id=image.composition_reference_id,
                seed=seed_value,
                status=ImageStatus.TEMPORARY.value,
                image_type=Type.EDITED.value,
                # Lineage tracking
                root_id=lineage_context['root_id'],
                parent_id=lineage_context['parent_id'],
                lineage_root_id=lineage_context['lineage_root_id'],  # NEW
                version_source_id=image.id,  # Created FROM this image
                # CHANGED: Inherit from parent (not None)
                version_number=image.version_number,
                lineage_version_number=None,
                is_base=False,  # Not a base version yet
                generation_batch_id=uuid.uuid4(),
            )
            db.add(image_obj)
            await db.flush()
            logger.info(
                f"Created database record: {image_obj.image_name} (ID: {image_obj.id})")

            # Generate presigned URLs for response
            response_url = generate_presigned_url(file_key)
            logger.info(f"Starting get_image_analysis")

            prompt = getattr(prompt_obj, 'enhanced_prompt', None)
            logger.info(f"Using enhanced prompt: {prompt}")

            preset_dict = getattr(prompt_obj, 'guidelines_applied', None)
            logger.debug(f"Guidelines applied (preset_dict): {preset_dict}")
            presets = extract_values(preset_dict)
            logger.info(f"Extracted presets for guardrails: {presets}")

            logger.info("Calling analyze_image API...")
            data = await analyze_image(
                image_url=response_url,
                presets=presets,
                prompt=prompt,
            )
            logger.info(
                f"Image analysis completed. Matched guardrails: {data.get('matched_guardrails')}, Missing guardrails: {data.get('missing_guardrails')}")
            logger.debug(f"Analysis API response: {data}")

            applied_guardrails = check_compliance(
                preset_dict=preset_dict,
                matched_guardrails=data["matched_guardrails"],
                missing_guardrails=data["missing_guardrails"],
            )
            logger.info(
                f"Compliance checked. Applied guardrails: {applied_guardrails}")

            compliance_score = calculate_compliance_ratio(
                data["matched_guardrails"], data["missing_guardrails"])
            logger.info(f"Compliance score calculated: {compliance_score}")
            result = {
                "id": image_obj.id,
                "image_url": response_url,
                "prompt": prompt,
                "analysis": {
                    "score": str(compliance_score),
                    "suggestion": data["suggestion"],
                    "suggestedPrompt": data["suggested_prompt"],
                    "guardrails": applied_guardrails,
                },
            }
            logger.info(
                f"Returning image analysis result for image_id={image.id}")
            logger.debug(f"Image analysis result: {result}")
            await db.commit()
            return result
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(
                f"An Exception Occurred in Generate Compliant API: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=settings.err_msg
            )

    async def generate_compliant_image_stream(self, db, image, prompt_obj, image_type, enhanced_prompt, access_token):
        """SSE generator with unified response format"""

        def send_response(type, status, code, percent, message, data) -> str:
            """Create standardized SSE response"""
            response = SSEResponse(
                type=type,
                status=status,
                code=code,
                percent=percent,
                message=message,
                data=data,
                timestamp=datetime.utcnow()
            )
            return response.to_sse_format()

        try:
            # STAGE 1: Submit to Adobe Firefly (15%)
            logger.info("Submitting to Adobe Firefly...")
            yield send_response(
                type=ResponseType.PROGRESS,
                status=ResponseStatus.PROCESSING,
                code=ResponseCode.PROGRESS,
                percent=15,
                message="Submitting request to Adobe Firefly...",
                data={"stage": "submit_adobe", "attempt": "1/1"}
            )

            if image_type == "photo":
                model = "image4_ultra"
                content_class = "photo"
                negative_prompt = "low quality, worst quality, blurry, out of focus, pixelated, jpeg artifacts, noise, grainy, distorted, compressed, bad anatomy, wrong anatomy, bad proportions, deformed body, malformed limbs, extra limbs, missing limbs, bad hands, deformed hands, extra fingers, missing fingers, fused fingers, twisted fingers, poorly drawn hands, asymmetrical face, misaligned features, deformed face, extra eyes, wrong skin tones, unrealistic textures, elongated body, long neck, disproportionate body, cluttered background, distracting background, random objects, floating objects, watermark, text, NSFW, gore, poor lighting, oversaturated, underexposed, unnatural appearance, mutation"
            elif image_type == "icon":
                model = "image4_custom"
                content_class = "art"
                negative_prompt = "multiple lines, broken lines, disconnected strokes, separate line segments, gaps in line, outline style, filled shapes, closed loops, individual elements, multiple strokes, traditional line art, sketchy lines, double lines, overlapping lines, thick outlines, cartoon style, detailed shading, hatching, cross-hatching, filled areas, solid fills, gradient fills, color fills, shadows, highlights, 3D effects, perspective depth, realistic rendering, photographic style, textured surfaces, pattern fills, background elements, decorative details, ornamental features, complex details, intricate patterns, multiple colors, color variations, blurred edges, rough sketches, hand-drawn imperfections, uneven lines, wavy lines, dotted lines, dashed lines, segmented paths, separate shapes, disconnected objects, isolated elements, layered strokes, composite drawings"

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": settings.ADOBE_CLIENT_ID,
                "Authorization": f"Bearer {access_token}",
                "x-model-version": model
            }

            payload = {
                "contentClass": content_class,
                "negativePrompt": negative_prompt,
                "numVariations": 1,
                "prompt": enhanced_prompt[:1024],
                "size": {"height": image.height, "width": image.width},
                "visualIntensity": 5,
                "seeds": [image.seed]
            }

            if image_type == "icon":
                payload.pop("visualIntensity")
                payload["customModelId"] = settings.ADOBE_CUSTOM_MODEL

            logger.info(
                f"Adobe Firefly payload: {json.dumps(payload, indent=2)}")

            # FIX #1: Use asyncio.to_thread for blocking request
            response = await asyncio.to_thread(
                requests.post,
                "https://firefly-api.adobe.io/v3/images/generate-async",
                headers=headers,
                data=json.dumps(payload),
                timeout=90
            )

            # FIX #2: Check status code before parsing
            if response.status_code != 202:
                logger.error(
                    f"Adobe API error: {response.status_code} - {response.text}")
                yield send_response(
                    type=ResponseType.ERROR,
                    status=ResponseStatus.ERROR,
                    code=ResponseCode.ERROR_ADOBE_API,
                    percent=0,
                    message=f"Adobe API error: {response.status_code}",
                    data={"error_response": response.text[:200]}
                )
                return

            res = response.json()
            status_url = res.get("statusUrl")

            if not status_url:
                logger.error("No statusUrl in Adobe response")
                yield send_response(
                    type=ResponseType.ERROR,
                    status=ResponseStatus.ERROR,
                    code=ResponseCode.ERROR_ADOBE_API,
                    percent=0,
                    message="Invalid Adobe response: no statusUrl",
                    data={}
                )
                return

            # STAGE 2: Poll Adobe (15-70%)
            logger.info("Polling Adobe Firefly...")
            max_retries = 60
            image_urls = []

            for attempt in range(max_retries):
                progress = 15 + (attempt / max_retries) * 55
                percentage_complete = int((attempt / max_retries) * 100)

                yield send_response(
                    type=ResponseType.PROGRESS,
                    status=ResponseStatus.PROCESSING,
                    code=ResponseCode.PROGRESS,
                    percent=int(progress),
                    message=f"Generating image... {percentage_complete}% complete",
                    data={
                        "stage": "generating_image",
                        "attempt": f"{attempt + 1}/{max_retries}",
                        "phase_progress": percentage_complete
                    }
                )

                # FIX #3: Wrap in asyncio.to_thread if get_image is sync
                image_fetch_res = await asyncio.to_thread(get_image, status_url, access_token)

                # FIX #6: Use .get() instead of direct access
                status_value = image_fetch_res.get("status")

                if not status_value:
                    logger.error("No status in get_image response")
                    yield send_response(
                        type=ResponseType.ERROR,
                        status=ResponseStatus.ERROR,
                        code=ResponseCode.ERROR_ADOBE_API,
                        percent=0,
                        message="Invalid Adobe status response",
                        data={}
                    )
                    return

                logger.info(
                    f"Attempt {attempt + 1}/{max_retries} - Status: {status_value}")

                if status_value == "succeeded":
                    image_urls = image_fetch_res.get(
                        "result", {}).get("outputs", [])

                    # FIX #4: Validate image_urls
                    if not image_urls:
                        logger.error("No outputs in Adobe response")
                        yield send_response(
                            type=ResponseType.ERROR,
                            status=ResponseStatus.ERROR,
                            code=ResponseCode.ERROR_ADOBE_API,
                            percent=0,
                            message="No image outputs from Adobe",
                            data={}
                        )
                        return

                    logger.info(
                        f"Image generation succeeded after {attempt + 1} attempts")
                    break

                elif status_value == "failed":
                    logger.error(f"Adobe generation failed: {image_fetch_res}")
                    yield send_response(
                        type=ResponseType.ERROR,
                        status=ResponseStatus.ERROR,
                        code=ResponseCode.ERROR_ADOBE_API,
                        percent=0,
                        message="Adobe Firefly image generation failed",
                        data={"error_details": str(image_fetch_res)[:200]}
                    )
                    return

                elif status_value not in ("pending", "running"):
                    logger.error(f"Unexpected status: {status_value}")
                    yield send_response(
                        type=ResponseType.ERROR,
                        status=ResponseStatus.ERROR,
                        code=ResponseCode.ERROR_ADOBE_API,
                        percent=0,
                        message=f"Unexpected Adobe status: {status_value}",
                        data={}
                    )
                    return

                await asyncio.sleep(2)
            else:
                logger.error("Image generation timed out")
                yield send_response(
                    type=ResponseType.ERROR,
                    status=ResponseStatus.ERROR,
                    code=ResponseCode.ERROR_TIMEOUT,
                    percent=0,
                    message="Image generation timed out after 120 seconds",
                    data={"max_retries": max_retries}
                )
                return

            # Extract image
            img_data = image_urls[0]
            generated_url = img_data.get("image", {}).get("url")
            seed_value = img_data.get("seed")

            # FIX #4: Validate URL exists
            if not generated_url:
                logger.error("No image URL in Adobe response")
                yield send_response(
                    type=ResponseType.ERROR,
                    status=ResponseStatus.ERROR,
                    code=ResponseCode.ERROR_ADOBE_API,
                    percent=0,
                    message="No image URL in Adobe response",
                    data={}
                )
                return

            # SEND IMAGE (75%)
            logger.info("Image generated, sending to user...")
            yield send_response(
                type=ResponseType.IMAGE,
                status=ResponseStatus.SUCCESS,
                code=ResponseCode.IMAGE_RECEIVED,
                percent=75,
                message="Generated image received",
                data={
                    "image_url": generated_url,
                    "seed": seed_value,
                    "size": f"{image.width}x{image.height}"
                }
            )
            # Download image
            logger.info(
                f"Downloading generated image from: {generated_url}")
            resp = await asyncio.to_thread(requests.get, generated_url, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Failed to download image: {resp.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to download generated image"
                )

            image_bytes = resp.content
            logger.info(f"Downloaded image size: {len(image_bytes)} bytes")

            # Generate S3 keys
            uid = uuid.uuid4().hex[:8]
            file_key = f"gilead/imagegen/editedimages/originals/image_{uid}.jpg"
            thumb_key = f"gilead/imagegen/editedimages/thumbnails/thumb_{uid}.webp"

            # Upload original to S3
            logger.info(f"Uploading to S3: {file_key}")
            upload_to_s3(io.BytesIO(image_bytes), file_key,
                         content_type="image/jpeg")

            # Generate and upload thumbnail
            logger.info("Generating thumbnail...")
            img = PILImage.open(io.BytesIO(image_bytes))
            img.thumbnail(THUMBNAIL_MAX_SIZE)
            thumb_buffer = io.BytesIO()
            img.save(thumb_buffer, format="WEBP", optimize=True, quality=100)
            thumb_buffer.seek(0)

            logger.info(f"Uploading thumbnail: {thumb_key}")
            upload_to_s3(thumb_buffer, thumb_key, content_type="image/webp")

            # CHANGED: Get lineage context
            lineage_context = get_root_and_parent(image)
            logger.info(f"Creating edited image with lineage: root_id={lineage_context['root_id']}, "
                        f"parent_id={lineage_context['parent_id']}, "
                        f"lineage_root_id={lineage_context['lineage_root_id']}, "
                        f"version_source_id={image.id}")

            image_name = await generate_next_image_name(db)

            # CHANGED: Create Image with lineage fields
            image_obj = Image(
                user_id=image.user_id,
                width=image.width,
                height=image.height,
                image_name=image_name,
                s3_object_key=file_key,
                s3_thumbnail_key=thumb_key,
                prompt_id=image.prompt_id,
                project_id=image.project_id,
                style_reference_id=image.style_reference_id,
                composition_reference_id=image.composition_reference_id,
                seed=seed_value,
                status=ImageStatus.TEMPORARY.value,
                image_type=Type.EDITED.value,

                # Lineage tracking
                root_id=lineage_context['root_id'],
                parent_id=lineage_context['parent_id'],
                lineage_root_id=lineage_context['lineage_root_id'],  # NEW

                version_source_id=image.id,  # Created FROM this image
                # CHANGED: Inherit from parent (not None)
                version_number=image.version_number,
                lineage_version_number=None,  # NEW: Will be assigned when saved

                is_base=False,  # Not a base version yet
                generation_batch_id=uuid.uuid4(),
            )

            db.add(image_obj)
            await db.flush()
            logger.info(
                f"Created database record: {image_obj.image_name} (ID: {image_obj.id})")

            # Generate presigned URLs for response
            response_url = generate_presigned_url(file_key)

            # STAGE 3: Analyze Compliance (80-92%)
            logger.info("Starting compliance analysis...")
            yield send_response(
                type=ResponseType.PROGRESS,
                status=ResponseStatus.PROCESSING,
                code=ResponseCode.PROGRESS,
                percent=80,
                message="Analyzing brand compliance with guardrails...",
                data={"stage": "analyzing_compliance",
                      "substage": "gpt_vision_analysis"}
            )

            # Analyze image
            prompt_text = getattr(prompt_obj, 'enhanced_prompt', None)
            preset_dict = getattr(prompt_obj, 'guidelines_applied', None)
            presets = extract_values(preset_dict)
            image_url = generate_presigned_url(image_obj.s3_object_key)

            data = await analyze_image(
                image_url=image_url,
                presets=presets,
                prompt=prompt_text,
            )

            # FIX #5: Validate required keys in analysis response
            required_analysis_keys = [
                'matched_guardrails', 'missing_guardrails', 'suggestion', 'suggested_prompt']
            for key in required_analysis_keys:
                if key not in data:
                    logger.error(f"Missing '{key}' in analysis response")
                    yield send_response(
                        type=ResponseType.ERROR,
                        status=ResponseStatus.ERROR,
                        code=ResponseCode.ERROR_GPT_ANALYSIS,
                        percent=0,
                        message=f"Analysis response missing required field: {key}",
                        data={}
                    )
                    return

            logger.info("Compliance analysis complete")
            yield send_response(
                type=ResponseType.PROGRESS,
                status=ResponseStatus.PROCESSING,
                code=ResponseCode.PROGRESS,
                percent=92,
                message="Compliance analysis complete",
                data={
                    "stage": "analyzing_compliance",
                    "matched_count": len(data.get('matched_guardrails', [])),
                    "missing_count": len(data.get('missing_guardrails', []))
                }
            )

            # STAGE 4: Calculate Score (95%)
            logger.info("Calculating compliance score...")
            yield send_response(
                type=ResponseType.PROGRESS,
                status=ResponseStatus.PROCESSING,
                code=ResponseCode.PROGRESS,
                percent=95,
                message="Calculating compliance score...",
                data={"stage": "calculating_score"}
            )

            applied_guardrails = check_compliance(
                preset_dict=preset_dict,
                matched_guardrails=data["matched_guardrails"],
                missing_guardrails=data["missing_guardrails"],
            )

            compliance_score = calculate_compliance_ratio(
                data["matched_guardrails"],
                data["missing_guardrails"]
            )

            # STAGE 5: Complete (100%)
            logger.info("Finalizing results...")
            result = {
                "id": image_obj.id,
                "image_url": image_url,
                "prompt": enhanced_prompt,
                "analysis": {
                    "score": str(compliance_score),
                    "suggestion": data["suggestion"],
                    "suggestedPrompt": data["suggested_prompt"],
                    "guardrails": applied_guardrails,
                },
            }

            await db.commit()

            yield send_response(
                type=ResponseType.RESULT,
                status=ResponseStatus.SUCCESS,
                code=ResponseCode.ANALYSIS_COMPLETE,
                percent=100,
                message="Image generation and analysis complete",
                data=result
            )

        except HTTPException as e:
            logger.error(f"HTTP Exception: {e.detail}")
            yield send_response(
                type=ResponseType.ERROR,
                status=ResponseStatus.ERROR,
                code=ResponseCode.ERROR_IMAGE_NOT_FOUND,
                percent=0,
                message=str(e.detail),
                data={"error_code": e.status_code}
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            yield send_response(
                type=ResponseType.ERROR,
                status=ResponseStatus.ERROR,
                code=ResponseCode.ERROR_GPT_ANALYSIS,
                percent=0,
                message="An unexpected error occurred",
                data={"error_details": str(e)[:200]}
            )

    async def image_details(self, db, id):
        try:
            logger.info(f"Fetching version history for image: {id}")
            result = await db.execute(
                select(Image).where(Image.id == id)
            )
            target_image = result.scalar_one_or_none()
            if not target_image:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Image not found."
                )
            lineage_root_id = target_image.lineage_root_id or target_image.id
            # Fetch all versions
            versions_result = await db.execute(
                select(Image)
                .where(
                    or_(
                        Image.id == lineage_root_id,
                        Image.lineage_root_id == lineage_root_id
                    ),
                    Image.status == ImageStatus.PERMANENT.value
                )
                .order_by(Image.lineage_version_number.asc())
            )
            all_versions = versions_result.scalars().all()
            if not all_versions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No versions found."
                )
            project_result = await db.execute(
                select(Project).where(
                    Project.id == target_image.project_id)
            )
            project = project_result.scalar_one_or_none()
            prompt = None
            if target_image.prompt_id:
                prompt_result = await db.execute(
                    select(Prompt).where(Prompt.id == target_image.prompt_id)
                )
                prompt = prompt_result.scalar_one_or_none()

            license_data = {}
            license_result = await db.execute(
                select(LicenseFile).where(LicenseFile.image_id == id)
            )
            license_record = license_result.scalar_one_or_none()

            if license_record:
                license_data = {
                    "license_start_date": (
                        license_record.license_start_date
                        if license_record.license_start_date else None
                    ),
                    "license_end_date": (
                        license_record.license_end_date
                        if license_record.license_end_date else None
                    ),
                    "version": "v1.0",
                    "permitted_countries": license_record.permitted_countries or []
                }
            versions_data = []
            for version in all_versions:
                is_current = (version.id == target_image.id)
                versions_data.append({
                    "id": str(version.id),
                    "display_id": version.image_name,
                    "url": generate_presigned_url(version.s3_object_key),
                    "thumbnail_url": generate_presigned_url(version.s3_thumbnail_key),
                    "version": format_version((version.lineage_version_number)),
                    "seed": version.seed,
                    "is_published": version.is_published,
                    "source": version.image_source.lower(),
                    "created_at": format_date(version.created_at.isoformat()),
                    "product": project.product.capitalize() if project else None,
                    "brand": project.country.capitalize() if project else None,
                    "prompt": prompt.prompt if prompt else None,
                    "is_current": is_current,
                    "license_data": license_data
                })
            return versions_data
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(
                f"An Exception Occurred in Image Details API: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=settings.err_msg
            )

    async def edit_project_metadata(self, db, project_id, project_name):
        try:
            project_query = await db.execute(
                select(Project).where(
                    Project.id == project_id,
                    Project.is_active == True
                )
            )
            project = project_query.scalar_one_or_none()
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Project with ID {project_id} not found."
                )
            project.name = project_name
            db.add(project)
            await db.commit()
            await db.refresh(project)

            logger.info(
                f"Project {project_id} renamed to '{project_name}' successfully.")
            return {
                "project_id": str(project.id),
                "project_name": project.name
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(
                f"An Exception Occurred in Edit Project API: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=settings.err_msg
            )

    async def reuse_component(self, db, source, image):
        try:
            prompt = None
            enhanced_prompt = None
            validate = False

            if source == "local":
                project_result = await db.execute(
                    select(Project).where(Project.id == image.project_id,
                                          Project.is_active == True)
                )
                project = project_result.scalar_one_or_none()
                if not project:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Project not found for image ID: {image.id}"
                    )
                product, country, type_ = project.product, project.country, project.type
                project_obj = Project(
                    name=type_.capitalize(),
                    product=product,
                    country=country,
                    type=type_,
                )
                width = image.width
                height = image.height
                prompt_id = image.prompt_id

                # Fetch prompt and check for preset_dict
                if prompt_id:
                    prompt_result = await db.execute(
                        select(Prompt).where(Prompt.id == prompt_id)
                    )
                    prompt_obj = prompt_result.scalar_one_or_none()
                    if prompt_obj:
                        prompt = prompt_obj.prompt
                        enhanced_prompt = prompt_obj.enhanced_prompt

                        # Check if prompt has preset_dict for validation
                        if prompt_obj.guidelines_applied:
                            validate = True
                            logger.info(
                                f"Reused image prompt has preset_dict - validation enabled")
                        else:
                            logger.info(
                                f"Reused image prompt has no preset_dict - validation disabled")

                file_key = image.s3_object_key
                thumb_key = image.s3_thumbnail_key
                image_type = Type.LIBRARY.value
                style_reference_id = image.style_reference_id
                composition_reference_id = image.composition_reference_id
                seed_value = image.seed
                image_source = ImageSource.LOCAL.value
            else:
                product, country, type_ = image.brand, image.geography, image.classification
                project_obj = Project(
                    name=type_.capitalize(),
                    product=product,
                    country=country,
                    type=type_,
                )
                # Fetch and validate image
                image_url = image.s3_url
                

                if not image_url or not image_url.startswith(("http://", "https://")):
                    logger.error(f"❗ Invalid image_url '{image_url}' — generating presigned S3 URL.")
                    image_url = generate_presigned_url(image_url)
                    logger.info(f"✔️ Corrected image_url → {image_url}")

                # -----------------------
                # DOWNLOAD IMAGE SAFELY
                # -----------------------
                async with httpx.AsyncClient(timeout=10) as client:
                    logger.info(f"📡 Downloading Veeva image from: {image_url}")
                    resp = await client.get(image_url)
                    resp.raise_for_status()
                    logger.info(f"📥 Download successful ({len(resp.content)} bytes)")
                    image_bytes = resp.content

                # Read image dimensions
                img = PILImage.open(io.BytesIO(image_bytes))
                width, height = img.size
                # Generate S3 keys
                uid = uuid.uuid4().hex[:8]
                file_key = f"gilead/imagegen/generatedimages/originals/image_{uid}.jpg"
                thumb_key = f"gilead/imagegen/generatedimages/thumbnails/thumb_{uid}.webp"
                # Upload original image
                upload_to_s3(io.BytesIO(image_bytes), file_key, content_type="image/jpeg")
                thumb = img.copy()
                thumb.thumbnail(THUMBNAIL_MAX_SIZE)
                thumb_buffer = io.BytesIO()
                thumb.save(thumb_buffer, format="WEBP",
                           optimize=True, quality=100)
                thumb_buffer.seek(0)
                upload_to_s3(thumb_buffer, thumb_key, content_type="image/webp")

                # Set prompt and enhanced_prompt from Veeva image
                prompt = image.title
                enhanced_prompt = image.title

                # Create prompt entry (Veeva images don't have preset_dict)
                prompt_obj = Prompt(
                    prompt=prompt,
                    enhanced_prompt=enhanced_prompt
                )
                db.add(prompt_obj)
                await db.flush()
                prompt_id = prompt_obj.id

                # Veeva images have no preset_dict, so validate remains False
                validate = False
                logger.info(
                    f"Veeva image - no preset_dict, validation disabled")

                image_type = Type.LIBRARY.value
                style_reference_id = None
                composition_reference_id = None
                seed_value = 0
                image_source = ImageSource.VEEVA.value

            db.add(project_obj)
            await db.flush()
            image_name = await generate_next_image_name(db)
            batch_id = uuid.uuid4()

            img_obj = Image(
                user_id="system",
                width=str(width),
                height=str(height),
                image_name=image_name,
                s3_object_key=file_key,
                s3_thumbnail_key=thumb_key,
                prompt_id=prompt_id,
                project_id=project_obj.id,
                style_reference_id=style_reference_id,
                composition_reference_id=composition_reference_id,
                seed=seed_value,
                status=ImageStatus.PERMANENT.value,
                image_type=image_type,
                parent_id=None,
                root_id=None,
                version_source_id=None,
                version_number=1,
                lineage_version_number=1,
                iteration_number=1,
                lineage_root_id=None,
                is_base=True,
                generation_batch_id=batch_id,
            )
            db.add(img_obj)
            await db.flush()

            img_obj.lineage_root_id = img_obj.id
            await db.commit()

            return {
                "iteration": img_obj.iteration_number,
                "iteration_type": image_type.lower(),
                "project_id": str(project_obj.id),
                "project_name": project_obj.name,
                "project_code": f"{project_obj.code:05d}",
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "parameters": {
                    "product": product,
                    "country": country,
                    "type": type_,
                },
                "images": [
                    {
                        "id": str(img_obj.id),
                        "display_id": img_obj.image_name,
                        "url": generate_presigned_url(file_key),
                        "thumbnail_url": generate_presigned_url(thumb_key),
                        "style_reference_url": await get_style_reference_url(db, img_obj.style_reference_id)
                        if img_obj.style_reference_id else None,
                        "composition_reference_url": await get_composition_reference_url(db, img_obj.composition_reference_id)
                        if img_obj.composition_reference_id else None,
                        "is_published": img_obj.is_published,
                        "source": image_source.lower(),
                        "has_history": False,
                        "can_validate": validate
                    }
                ],
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Unexpected error in reuse_component: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=settings.err_msg,
            )

    async def go_to_project(self, db, id, image):
        try:
            logger.info(f"Fetching project details for image: {image.id}")
            project_result = await db.execute(
                select(Project).where(Project.id == image.project_id,
                                      Project.is_active == True)
            )
            project = project_result.scalar_one_or_none()

            if not project:
                logger.error(f"Project not found for image {image.id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )

            prompt = None
            if image.prompt_id:
                prompt_result = await db.execute(
                    select(Prompt).where(Prompt.id == image.prompt_id)
                )
                prompt_obj = prompt_result.scalar_one_or_none()
                if prompt_obj:
                    prompt = prompt_obj.prompt

            iteration_number = image.iteration_number

            max_iteration_result = await db.execute(
                select(func.max(Image.iteration_number))
                .where(
                    Image.project_id == image.project_id,
                    Image.status == ImageStatus.PERMANENT.value,
                    Image.iteration_number.isnot(None)
                )
            )
            max_iteration = max_iteration_result.scalar()

            if max_iteration is None:
                max_iteration = iteration_number

            logger.info(
                f"Found project: {project.name}, iteration: {iteration_number}/{max_iteration}")

            return {
                "id": id,
                "iteration": iteration_number,
                "total": max_iteration,
                "prompt": prompt,
                "project_id": str(project.id),
                "project_name": project.name,
                "project_code": f"{project.code:05d}" if project.code is not None else None,
                "parameters": {
                    "product": project.product,
                    "country": project.country,
                    "type": project.type
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Exception in go_to_project: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch project details"
            )

    async def edit_project(self, db: AsyncSession, project_id: str, limit: int):
        try:
            logger.info(
                f"Fetching {limit} recent iterations for project: {project_id}")

            project_result = await db.execute(
                select(Project).where(
                    Project.id == project_id,
                    Project.is_active == True
                )
            )
            project = project_result.scalar_one_or_none()

            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )

            max_iteration_result = await db.execute(
                select(func.max(Image.iteration_number))
                .where(
                    Image.project_id == project_id,
                    Image.status == ImageStatus.PERMANENT.value,
                    Image.iteration_number.isnot(None)
                )
            )
            max_iteration = max_iteration_result.scalar()

            if max_iteration is None:
                logger.info(f"No iterations found for project {project_id}")
                return {
                    "project_id": str(project.id),
                    "project_name": project.name,
                    "project_code": f"{project.code:05d}" if project.code is not None else None,
                    "parameters": {
                        "product": project.product,
                        "country": project.country,
                        "type": project.type
                    },
                    "data": [],
                    "total": 0,
                    "returned": 0
                }

            start_iteration = max(1, max_iteration - limit + 1)
            logger.info(
                f"Fetching iterations {start_iteration} to {max_iteration} (total: {max_iteration})")

            iterations_result = await db.execute(
                select(
                    Image.iteration_number,
                    func.min(Image.created_at).label('created_at')
                )
                .where(
                    Image.project_id == project_id,
                    Image.status == ImageStatus.PERMANENT.value,
                    Image.iteration_number.isnot(None),
                    Image.iteration_number >= start_iteration
                )
                .group_by(Image.iteration_number)
                .order_by(Image.iteration_number.desc())
            )
            iterations = iterations_result.all()

            if not iterations:
                logger.info(
                    f"No iterations found in range {start_iteration}-{max_iteration}")
                return {
                    "project_id": str(project.id),
                    "project_name": project.name,
                    "project_code": f"{project.code:05d}" if project.code is not None else None,
                    "parameters": {
                        "product": project.product,
                        "country": project.country,
                        "type": project.type
                    },
                    "data": [],
                    "total": max_iteration,
                    "returned": 0
                }

            iteration_numbers = [it.iteration_number for it in iterations]

            all_images_result = await db.execute(
                select(Image).where(
                    Image.project_id == project_id,
                    Image.iteration_number.in_(iteration_numbers),
                    Image.status == ImageStatus.PERMANENT.value,
                    Image.is_base == True
                ).order_by(Image.iteration_number, Image.created_at)
            )
            all_images = all_images_result.scalars().all()

            images_by_iteration = {}
            for img in all_images:
                if img.iteration_number not in images_by_iteration:
                    images_by_iteration[img.iteration_number] = []
                images_by_iteration[img.iteration_number].append(img)

            logger.info(
                f"Batch fetched {len(all_images)} images across {len(iterations)} iterations")

            # Batch fetch style references
            style_ref_ids = list(
                set([img.style_reference_id for img in all_images if img.style_reference_id]))
            style_refs_map = {}

            if style_ref_ids:
                try:
                    style_refs_result = await db.execute(
                        select(StyleReference).where(
                            StyleReference.id.in_(style_ref_ids))
                    )
                    style_refs = style_refs_result.scalars().all()

                    for ref in style_refs:
                        try:
                            style_refs_map[ref.id] = generate_presigned_url(
                                ref.s3_object_key)
                        except ClientError as e:
                            logger.error(
                                f"S3 error for style ref {ref.id}: {e}")
                            style_refs_map[ref.id] = None

                    logger.info(
                        f"Batch fetched {len(style_refs_map)} style references")
                except Exception as e:
                    logger.error(f"Error batch fetching style references: {e}")

            # Batch fetch composition references
            comp_ref_ids = list(set(
                [img.composition_reference_id for img in all_images if img.composition_reference_id]))
            comp_refs_map = {}

            if comp_ref_ids:
                try:
                    comp_refs_result = await db.execute(
                        select(CompositionReference).where(
                            CompositionReference.id.in_(comp_ref_ids))
                    )
                    comp_refs = comp_refs_result.scalars().all()

                    for ref in comp_refs:
                        try:
                            comp_refs_map[ref.id] = generate_presigned_url(
                                ref.s3_object_key)
                        except ClientError as e:
                            logger.error(
                                f"S3 error for composition ref {ref.id}: {e}")
                            comp_refs_map[ref.id] = None

                    logger.info(
                        f"Batch fetched {len(comp_refs_map)} composition references")
                except Exception as e:
                    logger.error(
                        f"Error batch fetching composition references: {e}")

            # Batch fetch prompts for validation check
            prompt_ids = list(
                set([img.prompt_id for img in all_images if img.prompt_id]))
            prompt_validation_map = {}

            if prompt_ids:
                try:
                    prompts_result = await db.execute(
                        select(Prompt).where(Prompt.id.in_(prompt_ids))
                    )
                    prompts = prompts_result.scalars().all()

                    for prompt in prompts:
                        # Check if prompt has preset_dict
                        prompt_validation_map[prompt.id] = bool(
                            prompt.guidelines_applied)

                    logger.info(
                        f"Batch fetched {len(prompt_validation_map)} prompts for validation")
                except Exception as e:
                    logger.error(f"Error batch fetching prompts: {e}")

            # Batch fetch variation counts for has_history check
            all_image_ids = [img.id for img in all_images]

            variations_count_result = await db.execute(
                select(
                    Image.parent_id,
                    func.count(Image.id).label('variation_count')
                )
                .where(
                    Image.parent_id.in_(all_image_ids),
                    Image.image_type == Type.VARIATION.value,
                    Image.status == ImageStatus.PERMANENT.value
                )
                .group_by(Image.parent_id)
            )
            variations_count_map = {
                row.parent_id: row.variation_count for row in variations_count_result.all()}

            # Batch fetch lineage info
            lineage_roots = list(
                set([img.lineage_root_id for img in all_images if img.lineage_root_id]))
            lineage_version_counts = {}

            if lineage_roots:
                lineage_counts_result = await db.execute(
                    select(
                        Image.lineage_root_id,
                        func.max(Image.lineage_version_number).label(
                            'max_version')
                    )
                    .where(
                        Image.lineage_root_id.in_(lineage_roots),
                        Image.status == ImageStatus.PERMANENT.value
                    )
                    .group_by(Image.lineage_root_id)
                )
                lineage_version_counts = {
                    row.lineage_root_id: row.max_version for row in lineage_counts_result.all()}

            logger.info(f"Batch fetched variation counts and lineage info")

            # Helper function to determine has_history
            def check_has_history(img):
                has_variations = variations_count_map.get(img.id, 0) > 0
                has_ancestors = (img.lineage_version_number or 1) > 1
                max_version_in_lineage = lineage_version_counts.get(
                    img.lineage_root_id, img.lineage_version_number or 1)
                has_successors = (
                    img.lineage_version_number or 1) < max_version_in_lineage

                return has_variations or has_ancestors or has_successors

            # Helper function to check if image can be validated
            def check_can_validate(img):
                if img.prompt_id:
                    return prompt_validation_map.get(img.prompt_id, False)
                return False

            history = []

            for iteration in iterations:
                iteration_num = iteration.iteration_number
                images = images_by_iteration.get(iteration_num, [])

                if not images:
                    continue

                first_image = images[0]
                image_type = first_image.image_type

                if image_type == Type.VARIATION.value:
                    logger.debug(
                        f"Processing VARIATION iteration {iteration_num}")

                    first_variation = images[0]
                    parent = None
                    parent_prompt = None
                    parent_enhanced_prompt = None

                    if first_variation.parent_id:
                        parent_result = await db.execute(
                            select(Image).where(
                                or_(
                                    Image.id == first_variation.parent_id,
                                    Image.lineage_root_id == first_variation.parent_id
                                ),
                                Image.status == ImageStatus.PERMANENT.value,
                                Image.is_base == True
                            ).order_by(Image.lineage_version_number.desc())
                            .limit(1)
                        )
                        parent = parent_result.scalar_one_or_none()

                        if parent and parent.prompt_id:
                            prompt_result = await db.execute(
                                select(Prompt).where(
                                    Prompt.id == parent.prompt_id)
                            )
                            parent_prompt_obj = prompt_result.scalar_one_or_none()
                            if parent_prompt_obj:
                                parent_prompt = parent_prompt_obj.prompt
                                parent_enhanced_prompt = parent_prompt_obj.enhanced_prompt

                    variations_data = []
                    for var in images:
                        variations_data.append({
                            "id": str(var.id),
                            "display_id": var.image_name,
                            "url": generate_presigned_url(var.s3_object_key),
                            "thumbnail_url": generate_presigned_url(var.s3_thumbnail_key),
                            "is_published": var.is_published,
                            "source": var.image_source.lower(),
                            "style_reference_url": style_refs_map.get(var.style_reference_id),
                            "composition_reference_url": comp_refs_map.get(var.composition_reference_id),
                            "has_history": check_has_history(var),
                            "can_validate": check_can_validate(var)
                        })

                    variationrootimage = None
                    if parent:
                        variationrootimage = {
                            "id": str(parent.id),
                            "display_id": parent.image_name,
                            "url": generate_presigned_url(parent.s3_object_key),
                            "thumbnail_url": generate_presigned_url(parent.s3_thumbnail_key),
                            "is_published": parent.is_published,
                            "source": parent.image_source.lower(),
                            "style_reference_url": style_refs_map.get(parent.style_reference_id),
                            "composition_reference_url": comp_refs_map.get(parent.composition_reference_id),
                            "has_history": check_has_history(parent),
                            "can_validate": check_can_validate(parent)
                        }

                    history.append({
                        "iteration": iteration_num,
                        "iteration_type": "variation",
                        "project_id": str(project.id),
                        "prompt": parent_prompt,
                        "enhanced_prompt": parent_enhanced_prompt,
                        "images": variations_data,
                        "variationrootimage": variationrootimage
                    })

                elif image_type == Type.ORIGINAL.value:
                    logger.debug(
                        f"Processing ORIGINAL iteration {iteration_num}")

                    prompt = None
                    enhanced_prompt = None

                    if first_image.prompt_id:
                        prompt_result = await db.execute(
                            select(Prompt).where(
                                Prompt.id == first_image.prompt_id)
                        )
                        prompt_obj = prompt_result.scalar_one_or_none()
                        if prompt_obj:
                            prompt = prompt_obj.prompt
                            enhanced_prompt = prompt_obj.enhanced_prompt

                    images_data = []
                    for img in images:
                        images_data.append({
                            "id": str(img.id),
                            "display_id": img.image_name,
                            "url": generate_presigned_url(img.s3_object_key),
                            "thumbnail_url": generate_presigned_url(img.s3_thumbnail_key),
                            "is_published": img.is_published,
                            "source": img.image_source.lower(),
                            "style_reference_url": style_refs_map.get(img.style_reference_id),
                            "composition_reference_url": comp_refs_map.get(img.composition_reference_id),
                            "has_history": check_has_history(img),
                            "can_validate": check_can_validate(img)
                        })

                    history.append({
                        "iteration": iteration_num,
                        "iteration_type": "generation",
                        "project_id": str(project.id),
                        "prompt": prompt,
                        "enhanced_prompt": enhanced_prompt,
                        "images": images_data,
                        "variationrootimage": None
                    })

                elif image_type == Type.LIBRARY.value:
                    logger.debug(
                        f"Processing LIBRARY iteration {iteration_num}")

                    prompt = None
                    enhanced_prompt = None

                    if first_image.prompt_id:
                        prompt_result = await db.execute(
                            select(Prompt).where(
                                Prompt.id == first_image.prompt_id)
                        )
                        prompt_obj = prompt_result.scalar_one_or_none()
                        if prompt_obj:
                            prompt = prompt_obj.prompt
                            enhanced_prompt = prompt_obj.enhanced_prompt

                    images_data = []
                    for img in images:
                        images_data.append({
                            "id": str(img.id),
                            "display_id": img.image_name,
                            "url": generate_presigned_url(img.s3_object_key),
                            "thumbnail_url": generate_presigned_url(img.s3_thumbnail_key),
                            "is_published": img.is_published,
                            "source": img.image_source.lower(),
                            "style_reference_url": style_refs_map.get(img.style_reference_id),
                            "composition_reference_url": comp_refs_map.get(img.composition_reference_id),
                            "has_history": check_has_history(img),
                            "can_validate": check_can_validate(img)
                        })

                    history.append({
                        "iteration": iteration_num,
                        "iteration_type": "library",
                        "project_id": str(project.id),
                        "prompt": prompt,
                        "enhanced_prompt": enhanced_prompt,
                        "images": images_data,
                        "variationrootimage": None
                    })

                else:
                    logger.warning(
                        f"Unknown image type '{image_type}' for iteration {iteration_num}, skipping")
                    continue

            logger.info(
                f"Successfully built history with {len(history)} iterations")

            return {
                "project_id": str(project.id),
                "project_name": project.name,
                "project_code": f"{project.code:05d}" if project.code is not None else None,
                "parameters": {
                    "product": project.product,
                    "country": project.country,
                    "type": project.type
                },
                "data": history,
                "total": max_iteration,
                "returned": len(history)
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Exception in edit_project: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch recent iterations"
            )

    async def delete_project(self, db: AsyncSession, project_id: str):
        try:
            logger.info(f"Soft deleting project: {project_id}")
            project_result = await db.execute(
                select(Project).where(Project.id == project_id,
                                      Project.is_active == True)
            )
            project = project_result.scalar_one_or_none()

            if not project:
                logger.error(f"Project not found: {project_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )
            if not project.is_active:
                logger.warning(f"Project {project_id} is already deleted")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Project is already deleted"
                )
            project.is_active = False
            project.updated_at = datetime.utcnow()  # Track when deleted

            await db.commit()

            logger.info(f"Successfully soft deleted project: {project_id}")
            return {
                "message": "Project deleted successfully",

            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Exception in delete_project: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete project"
            )


imagegen = ImageGen()
