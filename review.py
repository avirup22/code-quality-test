import json
import traceback
import asyncio
from typing import Any, Dict, List, Tuple, Optional, Literal
from datetime import datetime, timezone
from datetime import date
from dateutil.relativedelta import relativedelta
from botocore.exceptions import BotoCoreError, ClientError
from fastapi.responses import StreamingResponse
from celery.result import AsyncResult
from app.repository.veeva_plus_license import VeevaPlusLicense
from fastapi import (APIRouter,Body, Depends, File, Form, HTTPException, Query,
                     UploadFile, status)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from loguru import logger
from redis.asyncio import Redis
from sqlalchemy import select, func, and_,update,desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from dateutil import parser as date_parser  # <-- helps parse ISO strings safely


from app.api.deps import (attach_param_ids, compute_file_hash,get_project_code,
                          validate_content_safety, validate_value)
from app.core.celery import celery

from app.core.config import settings
from app.database.db import get_db
from app.models import Country, Image, ImageType, Product, Project, Prompt,TherapeuticArea, VeevaDoc, LicenseFile,VeevaClassifications,VeevaBrand,VeevaAudience,VeevaCreativeAgency,VeevaSegment,VeevaUploader, VeevaTopics,PermittedChannels,PermittedCountries
from app.models.images import ImageType as Type
from app.repository import imagegen, utils
from app.repository.prompts import refine_prompt, restructure_prompt
from app.repository.utils import generate_presigned_url, generate_token,upload_to_s3, icon_prompt
from app.schemas import (DownloadImageResponse, EditImage,EditProject, EnrichPrompt,ProjectListResponse,DeleteProjectResponse,
                         FiltersResponse, GenerateCompliantImage,
                         GenerateImage, GenerateImageResponse,
                         ImageAnalysisResponse, ImageParametersResponse,
                         PromptSuggestions, ReferenceResponse,EditProjectResponse,
                         ReferenceResponseList, ReferenceUrl,
                         SaveVariationsRequest, VariationRequest, ReuseComponentResponse)
from app.tasks.celery_task import generate_image_task
from app.schemas import (DownloadImageResponse, EnrichPrompt, FiltersResponse,CountriesResponse,GoToProjectResponse,
                         GenerateCompliantImage, GenerateImage,
                         GenerateImageResponse, ImageAnalysisResponse,
                         ImageParametersResponse,ProjectHistoryResponse, PromptSuggestions, ReferenceResponse, ReferenceUrl,GenerateMediaResponse,IngestRequestSpecific,
                         ReferenceResponseList,TokenRequest,TokenResponse,IngestRequest,IngestResponse,ImageResult,SearchResult,SearchRequest,ProjectSearchRequest,ProjectResult,GenerateImageResponse, VariationResponse, SaveVariationsResponse, ImageAsset,ImageVersionResponse,ImageGroupDTO)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from app.core.veeva import VeevaHelper
from openai import AsyncOpenAI, OpenAIError


async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)



router = APIRouter()

embeddings = OpenAIEmbeddings(model=settings.EMBED_MODEL)
vectorstore = PGVector(
    connection_string=settings.SQLALCHEMY_DATABASE_SSL_URI,
    collection_name="veeva_docs", 
    embedding_function=embeddings,
    use_jsonb=True,  # important to remove warnings

) 
veeva = VeevaHelper()

redis = Redis.from_url(settings.REDIS_URL)


@router.get("/imageFilters", response_model=FiltersResponse)
async def get_image_filters(db: AsyncSession = Depends(get_db)):
    try:
        logger.info("Processing Filters API GET request.")
        cached = await redis.get("image_filters_cache")
        if cached:
            logger.info("Returning filters from Redis cache.")
            response = json.loads(cached)
            return response
        response = await imagegen.get_filters(db)
        logger.info(f"Filters response: {response}")
        await redis.set("image_filters_cache", json.dumps(response), ex=43200)
        return response
    except HTTPException as e:
        logger.error(f"HTTPException in helper: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"An Exception Occurred in Filters API: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )


@router.get("/projectFilters", response_model=FiltersResponse)
async def get_project_filters(db: AsyncSession = Depends(get_db)):
    try:
        logger.info("Processing Filters API GET request.")
        cached = await redis.get("projects_filters_cache")
        if cached:
            logger.info("Returning filters from Redis cache.")
            response = json.loads(cached)
            return response
        response = await imagegen.get_filters(db)
        logger.info(f"Filters response: {response}")
        await redis.set("projects_filters_cache", json.dumps(response), ex=43200)
        return response
    except HTTPException as e:
        logger.error(f"HTTPException in helper: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"An Exception Occurred in Filters API: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )


@router.get("/countries" , response_model=CountriesResponse)
async def get_countries(db: AsyncSession = Depends(get_db)):
    try:
        logger.info("Processing Filters API GET request.")
        cached = await redis.get("country_cache")
        if cached:
            logger.info("Returning filters from Redis cache.")
            response = json.loads(cached)
            return response
        response = await imagegen.get_countries(db)
        logger.info(f"Filters response: {response}")
        await redis.set("country_cache", json.dumps(response), ex=86400)
        return response
    except HTTPException as e:
        logger.error(f"HTTPException in helper: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"An Exception Occurred in Filters API: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )

@router.get(
    "/imageParameters",
    response_model=ImageParametersResponse,
)
async def get_image_parameters(
    db: Session = Depends(get_db),
):
    try:
        logger.info("Processing Image Parameters API GET request.")
        cached = await redis.get("image_parameters_cache")
        if cached:
            logger.info("Returning Image Parameters from cache.")
            response = json.loads(cached)
            return response
        response = await imagegen.get_image_dimensions(db)
        logger.info(f"Image Parmeters response: {response}")
        response_serializable = [
            {**item, "id": str(item["id"])} for item in response
        ]
        await redis.set(
            "image_parameters_cache",
            json.dumps(response_serializable),
            ex=43200
        )
        return response_serializable
    except HTTPException:
        logger.warning(f"HTTPException Raised.")
        raise
    except Exception as e:
        logger.info(traceback.format_exc())
        logger.error(
            f"An Exception Occurred in Image Parameters API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_200_OK,
            detail=settings.err_msg
        )


@router.post(
    "/refinePrompt", status_code=status.HTTP_200_OK
)
async def prompt_refinement(
    prompt: str = Query(...),
    type: str = Query("photo", pattern="^(photo|icon)$")
    
):
    try:
        if not prompt or len(prompt) == 0:
            logger.error("Prompt Should not be Empty.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt should not be empty.",
            )
        if await validate_content_safety(prompt):
            logger.error("Prompt contains sensitive content.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt contains sensitive content.",
            )
        logger.info("Prompt and Suggestion Validated.")
        if type == "photo":
            response = await refine_prompt(prompt)
        elif type == "icon":
            response = await icon_prompt(prompt)
        return response
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An Exception Occurred in Refine Prompt API: {str(e)}")
        logger.error(f"Something went wrong:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.concept_err_msg,
        )


@router.post("/generateMedia")#,response_model=GenerateMediaResponse)
async def generate_media(payload: GenerateImage, db: AsyncSession = Depends(get_db)):
    try:
        prompt = (payload.prompt or "").strip()
        logger.info(f"Validating prompt: {prompt}")

        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt cannot be empty."
            )

        if await validate_content_safety(prompt):
            logger.warning("Prompt contains sensitive content.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt contains sensitive content."
            )

        parameters = payload.parameters or {}
        project_id = payload.project_id

        def get_param(key):
            if hasattr(parameters, key):
                return getattr(parameters, key)
            elif isinstance(parameters, dict):
                return parameters.get(key)
            return None

        product = get_param("product")
        country = get_param("country")
        image_type = get_param("type")

        if not image_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image type is required."
            )
            
        if project_id:
            result = await db.execute(
                select(Project).where(Project.id == project_id)
            )
            project_record = result.scalar_one_or_none()
            project_name = project_record.name
            project_code = f"{project_record.code:05d}" if project_record.code is not None else None

            if not project_record:
                logger.warning(f"Invalid project ID: {project_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Invalid Project ID."
                )

            if product and project_record.product != product:
                logger.warning(f"Product mismatch for project {project_id}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"This project is configured for product '{project_record.product}'"
                )

            if country and project_record.country != country:
                logger.warning(f"Country mismatch for project {project_id}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"This project is configured for country '{project_record.country}'"
                )
        else:
            logger.info(
                f"Creating new project with product={product}, country={country}")
            project_obj = Project(
                name = image_type.capitalize(),
                product=product,
                country=country,
                type=image_type
            )
            db.add(project_obj)
            await db.flush()
            project_id = project_obj.id
            project_name = project_obj.name
            project_code = f"{project_obj.code:05d}"

            # CRITICAL: Commit before queuing Celery task
            await db.commit()
            logger.info(f"Project committed: {project_id}")

        if product:
            result = await db.execute(
                select(Product).where(Product.key == product)
            )
            if not result.scalar_one_or_none():
                logger.warning(f"Product not found: {product}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Product not found."
                )

        if country:
            result = await db.execute(
                select(Country).where(Country.key == country)
            )
            if not result.scalar_one_or_none():
                logger.warning(f"Country not found: {country}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Market not found."
                )

        try:
            image_type = getattr(parameters, "type",
                                 None) or parameters.get("type")
            if not image_type:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Image type is required."
                )
        except ValueError:
            logger.warning(f"Invalid image type: {image_type}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image type not found."
            )

        logger.info("All validations passed. Triggering image generation.")

        # Queue Celery task (now project_id is guaranteed to exist)
        data = jsonable_encoder(payload)
        parameters_dict = parameters.model_dump() if hasattr(
            parameters, 'model_dump') else dict(parameters)
        
        task = generate_image_task.delay(
            data, prompt, parameters_dict, project_id)
        logger.info(f"Generate image task queued with Task ID: {task.id}")

        return {
            "task_id": task.id,
            "project_id": project_id,
            "project_name": project_name,
            "project_code": project_code
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled exception in generate_media endpoint:")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )
        
@router.get("/generateMedia/status/{task_id}")
async def get_generate_media_status(task_id: str):
    try:
        result = AsyncResult(task_id, app=celery)
        if result.state == "PENDING":
            return {"status": "queued", "data": {}}
        elif result.state == "STARTED":
            return {"status": "running", "data": {}}
        elif result.state == "SUCCESS":
            task_result = result.result
            logger.info(f"Result: {task_result}")
            logger.info(
                f"Keys: {task_result.keys() if isinstance(task_result, dict) else 'Not a dict'}")
            return task_result
        elif result.state == "FAILURE":
            error_message = str(result.result.get("data", {}).get("message", result.info)) \
                if isinstance(result.result, dict) else str(result.info)
            return {"status": "error", "data": {"message": error_message}}
        else:
            return {"status": result.state, "data": {}}
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"An exception occurred in Enrich Prompt API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )


@router.post("/generateVariation", response_model=VariationResponse)
async def generate_variation(
    payload: VariationRequest,
    db: Session = Depends(get_db),
):
    try:
        id = payload.id
        number_of_variations = payload.number_of_variations
        logger.info("Processing Generate Variations API POST Request.")
        result = await db.execute(select(Image).filter(Image.id == id))
        image = result.scalar_one_or_none()
        if not image:
            logger.error(f"Image with id {id} not found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested image could not be found."
            )
        response = await imagegen.generate_image_variations(db, number_of_variations, image)
        return response
    except HTTPException:
        logger.warning(f"HTTPException Raised.")
        raise
    except Exception as e:
        logger.error(
            f"An Exception Occurred in Generate Variations API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )


@router.post("/saveVariations", response_model=SaveVariationsResponse)
async def save_variations(
    payload: SaveVariationsRequest,
    db: Session = Depends(get_db),
):
    try:
        data = jsonable_encoder(payload)
        image_ids = data.get("ids")
        logger.info(
            f"Processing Save Variations API POST Request for {len(image_ids)} images.")
        response = await imagegen.save_image_variations(db, image_ids)
        return response
    except HTTPException:
        logger.warning(f"HTTPException Raised.")
        raise
    except Exception as e:
        logger.error(f"An Exception Occurred in Save Variations API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )


@router.get("/styleReference", response_model=ReferenceResponseList)
async def get_style_reference(db: Session = Depends(get_db)):
    try:
        logger.info("Processing Style Reference API GET request.")
        cached = await redis.get("style_reference_cache")
        if cached:
            logger.info("Returning Style Reference from Redis cache.")
            response = json.loads(cached)
            return response
        result = await imagegen.get_style_reference(db)
        response = ReferenceResponseList(root=result)
        await redis.set("style_reference_cache", json.dumps(response.model_dump()), ex=3300)
        return response
    except HTTPException:
        logger.warning(f"HTTPException Raised.")
        raise
    except Exception as e:
        logger.error(
            f"An exception occured in GET Style Reference API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_200_OK,
            detail=settings.err_msg
        )


@router.get("/compositionReference", response_model=ReferenceResponseList)
async def get_composition_reference(db: Session = Depends(get_db)):
    try:
        logger.info("Processing Composition Reference API GET request.")
        cached = await redis.get("composition_reference_cache")
        if cached:
            logger.info("Returning Composition Reference from Redis cache.")
            response = json.loads(cached)
            return response
        result = await imagegen.get_composition_reference(db)
        response = ReferenceResponseList(result)
        await redis.set("composition_reference_cache", json.dumps(response.model_dump()), ex=3300)
        return response
    except HTTPException:
        logger.warning(f"HTTPException Raised.")
        raise
    except Exception as e:
        logger.error(
            f"An exception occured in GET Composition Reference API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_200_OK,
            detail=settings.err_msg
        )


@router.post("/fileReference", response_model=ReferenceUrl)
async def upload_reference(
    id: Optional[str] = Query(
        None, description="Existing Image ID (optional)"),
    file: Optional[UploadFile] = File(
        None, description="Upload a new reference file"),
    type: str = Query("composition", pattern="^(style|composition)$"),
    db: AsyncSession = Depends(get_db),
):
    ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
    logger.info(f"{type.capitalize()} Reference API - POST Request")
    try:
        if (id and file) or (not id and not file):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input. Provide only one source."
            )
        if file:
            filename = file.filename.lower()
            if not filename.endswith(ALLOWED_EXTENSIONS):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid file type. Only .jpg, .jpeg, and .png files are allowed."
                )
            if file.content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid content type. Only image/jpeg and image/png are allowed."
                )
        record = None
        if id:
            result = await db.execute(select(Image).filter(Image.id == id))
            record = result.scalar_one_or_none()
            if not record:
                logger.error(f"Image not found for ID: {id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Image not found."
                )
        response_data = await imagegen.create_reference(db, record=record, file=file, type=type)
        return ReferenceUrl(**response_data)
    except HTTPException as http_err:
        logger.warning(f"HTTPException: {http_err.detail}")
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.exception(f"Unexpected error in {type} Reference API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )


@router.get(
    "/download/{id}",
    response_model=DownloadImageResponse,
)
async def download_image(id: str, db: Session = Depends(get_db)):
    try:
        logger.info(f"Download request received for image ID: {id}")
        result = await db.execute(select(Image).filter(Image.id == id))
        image = result.scalar_one_or_none()
        if not image:
            logger.error(f"Image with id {id} not found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested image could not be found."
            )
        try:
            logger.info(
                f"Image with ID {id} found. Generating presigned URL...")
            url = generate_presigned_url(image.s3_file_key)
            logger.info(
                f"Presigned URL generated successfully for image ID: {id}")
        except (BotoCoreError, ClientError) as e:
            logger.error(
                f"Failed to generate presigned URL for image {id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to generate download URL. Please try again later."
            )
        return {
            "id": image.id,
            "url": url
        }
    except HTTPException:
        logger.warning(f"HTTPException Raised.")
        raise
    except Exception as e:
        logger.exception(
            f"Unexpected error during image download for id {id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg,
        )


@router.get(
    "/suggestions",
    response_model=PromptSuggestions,
)
async def prompt_suggestions(
    type: str = Query("photo", pattern="^(photo|icon)$")
):
    try:
        logger.info(f"Request for prompt suggetsions.")
        if type == "photo":
            response = [
                "Diverse group of adults, standing close and interacting; warm smiles and supportive gestures; urban plaza, candid, photorealistic.",
                "Single confident person, three-quarter portrait, hopeful expression; shallow depth of field, bright natural light, photorealistic.",
                "Small group celebrating, raised hands, subtle confetti in background; candid moment, wide-angle, natural lighting, photorealistic.",
                "Young couple cooking together in a bright kitchen; candid, warm mood, naturalistic lighting, photorealistic.",
                "Person reading on a cozy sofa with a warm throw; soft sunlight from window, candid, lifestyle photo, photorealistic.",
                "Friends walking dogs along a tree-lined path; smiling, casual clothing, photorealistic.",
                "Doctor and patient in consultation, attentive body language; calm clinic room, natural lighting, professional but warm, photorealistic.",
                "Nurse demonstrating inhaler technique to adult patient; close-up of hands and device, instructional but non-promotional, photorealistic.",
                "Community health worker talking with senior at kitchen table; empathetic expressions, candid, photorealistic.",
                "Two scientists in lab coats discussing data at a workstation; candid interaction, clear lab environment, photorealistic.",
                "Researcher looking through microscope, focused expression; shallow depth, authentic lab equipment in background, photorealistic.",
                "Gloved hands holding a petri dish with visible culture, close-up; clinical, respectful composition, photorealistic.",
                "Caregiver gently helping older adult stand; warm lighting, compassionate expressions, home setting, photorealistic.",
                "Two friends embracing on a park bench, supportive moment; candid, soft focus, photorealistic.",
                "Family member holding hands with patient in living room; intimate, respectful framing, photorealistic.",
                "Volunteers setting up a community health booth; collaborative action, candid, daytime outdoor fair, photorealistic.",
                "Small group listening to community speaker, engaged expressions; park setting, natural lighting, photorealistic.",
                "Mobile clinic team greeting local residents at doorway; helpful gestures, friendly tone, photorealistic.",
                "Multi-generation family portrait in backyard, candid laughter; natural light, warm colors, photorealistic.",
                "Parent and teen sharing coffee and conversation on couch; relaxed, authentic moment, photorealistic.",
                "Sibling pair hugging; close-up, tender expression, photorealistic.",
                "Small group cycling together on a suburban path; energetic but natural, candid action shot, photorealistic.",
                "People practising gentle yoga in a sunlit studio; calm expressions, photorealistic.",
                "Morning walk with friends on beach boardwalk; candid, fresh atmosphere, photorealistic.",
                "Team meeting in modern office, collaboration around laptop; diverse participants, candid, photorealistic.",
                "Technician inspecting machinery on factory floor; safety gear, focused action, photorealistic.",
                "Healthcare admin at reception desk helping patient; welcoming expression, natural light, photorealistic.",
                "Adult at home organizing daily medication into a pillbox (no branded packaging); focused, calm, photorealistic.",
                "Person making a phone call for support while sitting at kitchen table; candid, reassuring mood, photorealistic.",
                "Individual tracking health data on tablet (screen blurred to avoid text); casual, modern setting, photorealistic.",
                "Patient smiling during a video call with clinician on tablet; warm room lighting, candid, photorealistic.",
                "Clinician reviewing notes while talking to patient remotely; focused, home-office background, photorealistic.",
                "Caregiver guiding elderly relative through a tablet call; supportive, candid, photorealistic.",
                "Portrait trio of diverse adults looking confidently at camera; neutral background, natural lighting, photorealistic.",
                "Community collage: people of different ages and backgrounds sharing a table; candid, inclusive, photorealistic.",
                "Solo portrait of older adult with visible mobility aid; dignified, warm lighting, photorealistic.",
                "Engineer in protective gear inspecting sterile production equipment; clear factory setting, photorealistic.",
                "Small team adjusting controls on a clean processing line; focused teamwork, photorealistic.",
                "Operator in lab-style clean suit posing confidently near stainless steel tanks; professional, photorealistic.",
                "Instructor demonstrating a medical device (device blurred / non-branded) to learners; classroom lab setting, photorealistic.",
                "Group watching a presentation with attentive expressions; modern training room, candid, photorealistic.",
                "Student practicing bandaging technique on mannequin; hands-on, realistic instruction, photorealistic.",
                "Person gazing out window with soft contemplative expression; warm interior light, shallow depth, photorealistic.",
                "Supportive hand on shoulder in a softly lit room; close-up detail shot, intimate but respectful, photorealistic.",
                "Two friends sitting quietly on a park bench at dusk; subtle, reflective mood, photorealistic."
            ]
        elif type == "icon":
            response = [
                "Medicine bottle with pills, continuous line drawing, minimal red outline",
                "Stethoscope icon, continuous line drawing, minimal red outline",
                "Syringe with needle, continuous line drawing, minimal red outline",
                "Heart with ECG pulse line, continuous line drawing, minimal red outline",
                "Capsule pill split in half, continuous line drawing, minimal red outline",
                "Medical cross symbol, continuous line drawing, minimal red outline",
                "Thermometer icon, continuous line drawing, minimal red outline",
                "Blood drop icon, continuous line drawing, minimal red outline",
                "DNA helix double strand, continuous line drawing, minimal red outline",
                "Microscope icon, continuous line drawing, minimal red outline",
                "Bandage or adhesive strip, continuous line drawing, minimal red outline",
                "Mortar and pestle pharmacy tool, continuous line drawing, minimal red outline",
                "IV drip bag with tube, continuous line drawing, minimal red outline",
                "Medical clipboard with document, continuous line drawing, minimal red outline",
                "Brain icon, continuous line drawing, minimal red outline"
            ]
        return response
    except Exception as e:
        logger.exception(
            f"Unexpected error during prompt suggestions: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg,
        )


@router.post("/editImage",response_model=ImageAsset)
async def edit_image(
    payload: EditImage,
    db: Session = Depends(get_db)
):
    try:
        id = payload.id
        result = await db.execute(select(Image).filter(Image.id == id))
        image = result.scalar_one_or_none()
        if not image:
            logger.error(f"Image with id {id} not found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested image could not be found."
            )
        prompt = payload.prompt
        if await validate_content_safety(prompt):
            logger.warning("Prompt contains sensitive content.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt contains sensitive content."
            )
        project_record = await db.execute(select(Project).filter(Project.id == image.project_id))
        project = project_record.scalar_one_or_none()
        if not project:
            logger.error(f"Project with id {image.project_id} not found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image is not associated with any project."
            )
        image_type = project.type
        style_strength = payload.style_reference
        structure_strength = payload.composition_reference
        response = await imagegen.edit_image(db, image, prompt, style_strength, structure_strength, image_type)
        return response
    except HTTPException:
        logger.warning(f"HTTPException Raised.")
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"An Exception Occurred in Edit Image API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )


@router.post("/saveEditedImage", response_model=ImageAsset)
async def save_edited_image(
    id: str = Query(...),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Processing Save Edited Image request for ID: {id}")
        result = await db.execute(select(Image).filter(Image.id == id))
        image = result.scalar_one_or_none()
        if not image:
            logger.error(f"Image with id {id} not found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested image could not be found."
            )
        response = await imagegen.save_edited_image(db, id)
        return response
    except HTTPException:
        logger.warning("HTTPException raised in save_edited_image endpoint")
        raise
    except Exception as e:
        logger.error(f"Exception in save_edited_image endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )
        
        
@router.get("/imageDetails")#, response_model=ImageAsset)
async def get_image_details(
    id: str = Query(...),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Processing Save Edited Image request for ID: {id}")
        result = await db.execute(select(Image).filter(Image.id == id))
        image = result.scalar_one_or_none()
        if not image:
            logger.error(f"Image with id {id} not found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested image could not be found."
            )
        response = await imagegen.image_details(db, id)
        return response
    except HTTPException:
        logger.warning("HTTPException raised in save_edited_image endpoint")
        raise
    except Exception as e:
        logger.error(f"Exception in save_edited_image endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )
        
@router.get("/analyzeImage", response_model=ImageAnalysisResponse)
async def image_analysis(
        id: str = Query(...),
        db: Session = Depends(get_db)):
    try:
        logger.info(f"Processing Image Analysis API GET request for id={id}.")
        result = await db.execute(select(Image).filter(Image.id == id))
        image_obj = result.scalar_one_or_none()
        if not image_obj:
            logger.error(f"Image with id {id} not found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested image could not be found."
            )

        logger.debug(
            f"Found image for analysis: id={getattr(image_obj, 'id', None)}, prompt_id={getattr(image_obj, 'prompt_id', None)}"
        )

        prompt_result = await db.execute(select(Prompt).filter(Prompt.id == image_obj.prompt_id))
        prompt_obj = prompt_result.scalar_one_or_none()

        if not prompt_obj:
            logger.error(
                f"Prompt with id {getattr(image_obj, 'prompt_id', None)} not found for image {id}."
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prompt for the requested image could not be found."
            )

        if not getattr(prompt_obj, "guidelines_applied", False):
            logger.error(
                f"Image with id {id} missing guideline configuration (guidelines_applied={getattr(prompt_obj, 'guidelines_applied', None)})."
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image missing guideline configuration."
            )
            
        
        project_result = await db.execute(select(Project).filter(Project.id == image_obj.project_id))
        project_obj = project_result.scalar_one_or_none()
        
        if not project_obj:
            logger.error(
                f"Project with id {getattr(image_obj, 'project_id', None)} not found for image {id}."
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project for the requested image could not be found."
            )
        
        image_type = project_obj.type

        logger.info(f"Requesting image analysis for image id={id}")
        response = await imagegen.get_image_analysis(image_obj,image_type, prompt_obj)
        logger.info(f"Image analysis completed for id={id}")
        return response
    except HTTPException:
        logger.warning("HTTPException raised in image_analysis endpoint")
        raise
    except Exception as e:
        logger.exception(f"An exception occurred in GET Image Analysis API for id={id}: {str(e)}")
        # return the schema-shaped error response so the endpoint matches the response_model
        return ImageAnalysisResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            result=None,
            message=settings.err_msg,
        )

@router.get("/imageFamily",response_model=ImageVersionResponse)
async def get_image_family(
    id: str = Query(..., description="ID of any image"),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Processing Get Image Family request for ID: {id}")
        # Just pass the id string directly
        response = await imagegen.get_image_family(db, id)
        return response

    except HTTPException:
        logger.warning("HTTPException raised in get_image_family endpoint")
        raise
    except Exception as e:
        logger.error(f"Exception in get_image_family endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=settings.err_msg
        )


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    product: Optional[str] = Query(None, description="Filter by product"),
    country: Optional[str] = Query(None, description="Filter by country"),
    type: Optional[str] = Query(None, description="Filter by project type"),
    search: Optional[str] = Query(None, description="Search by prompt text"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        20, ge=1, le=100, description="Number of records to return"),
    db: AsyncSession = Depends(get_db)
):
    try:
        result = await imagegen.list_all_projects(
            db=db,
            product=product,
            country=country,
            type=type,
            search=search,
            skip=skip,
            limit=limit
        )
        return ProjectListResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch projects: {str(e)}"
        )


@router.get("/projectHistory", response_model=ProjectHistoryResponse)
async def get_project_history(
    project_id: str,
    skip: int = Query(0, ge=0, description="Number of iterations to skip"),
    limit: int = Query(
        5, ge=1, le=50, description="Number of iterations to return"),
    db: AsyncSession = Depends(get_db)
):
    try:
        result = await imagegen.get_project_history(db, project_id, skip, limit)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch project history"
        )

@router.post("/editProjectMetadata",response_model=EditProject)
async def edit_project_metadata(
    payload: EditProject,
    db: AsyncSession = Depends(get_db)
):
    try:
        project_id = payload.project_id
        project_name = payload.project_name
        if not len(project_name.strip()):
            logger.error("Project name is empty.")
            raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project name cannot be blank.") 
        result = await imagegen.edit_project_metadata(db, project_id, project_name)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch project history"
        )
        
        
@router.get("/editProject", response_model=EditProjectResponse)
async def edit_project(
    project_id: str,
    limit: int = Query(
        5, ge=1, le=100, description="Number of iterations to return"),
    db: AsyncSession = Depends(get_db)
):
    try:
        result = await imagegen.edit_project(db, project_id, limit)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch project history"
        )
        

@router.post("/reuseComponent", response_model=ReuseComponentResponse)
async def reuse_component(
    id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        result = await db.execute(select(Image).filter(Image.id == id))
        image = result.scalar_one_or_none()
        source = "local"
        if not image:
            result = await db.execute(select(VeevaDoc).filter(VeevaDoc.id == id))
            image = result.scalar_one_or_none()
            source = "veeva"

        if not image:
            logger.error(f"Image with id {id} not found in both images and veeva_doc tables.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested image could not be found."
            )
        response = await imagegen.reuse_component(db, source, image)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch project history"
        )
        

@router.get("/goToProject", response_model=GoToProjectResponse)
async def go_to_project(
    id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        result = await db.execute(select(Image).filter(Image.id == id))
        image = result.scalar_one_or_none()
        if not image:
            logger.error(
                f"Image with id {id} not found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested image could not be found."
            )
        response = await imagegen.go_to_project(db, id, image)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch project"
        )
        
@router.delete("/deleteProject", response_model=DeleteProjectResponse)
async def delete_project(
    project_id: str = Query(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        result = await imagegen.delete_project(db, project_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete project"
        )


@router.post("/generateCompliantImage")
async def generate_compliant_image(
        payload: str = Body(..., media_type="application/json"),
        db: Session = Depends(get_db)):
    try:
        logger.info("Compliant Image API POST request.")
        try:
            payload_data = json.loads(payload)
            payload = GenerateCompliantImage(**payload_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON string: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid JSON format")
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))
        
        id = payload.id
        # Validate image exists
        result = await db.execute(select(Image).filter(Image.id == id))
        image = result.scalar_one_or_none()
        if not image:
            logger.error(f"Image with id {id} not found.")
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Validate prompt exists
        prompt_result = await db.execute(select(Prompt).filter(Prompt.id == image.prompt_id))
        prompt_obj = prompt_result.scalar_one_or_none()
        if not prompt_obj or not getattr(prompt_obj, "guidelines_applied", False):
            logger.error(f"Guidelines not configured for image {id}.")
            raise HTTPException(
                status_code=404, detail="Guidelines not configured")
        
        # Validate content safety
        prompt = payload.prompt
        # if validate_content_safety(prompt):
        #     logger.error("Prompt contains sensitive content.")
        #     raise HTTPException(
        #         status_code=400, detail="Prompt contains sensitive content")
        
        # SKIP stages 1-2: Close DB session early to avoid holding connections
        project_result = await db.execute(select(Project).filter(Project.id == image.project_id))
        project_obj = project_result.scalar_one_or_none()
        
        if not project_obj:
            logger.error(
                f"Project with id {getattr(image, 'project_id', None)} not found for image {id}."
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project for the requested image could not be found."
            )
        
        image_type = project_obj.type
        enhanced_prompt = await restructure_prompt(prompt=prompt)
        access_token = await generate_token()
                
        # Return streaming response
        return StreamingResponse(
            imagegen.generate_compliant_image_stream(
                db=db,
                image=image,
                prompt_obj=prompt_obj,
                image_type=image_type,
                enhanced_prompt=enhanced_prompt,
                access_token=access_token
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Exception in generate compliant image endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=settings.err_msg)


@router.post("/auth")
def auth_api():
    """Return a Veeva session id."""
    try:
        session_id = veeva.auth()
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/ingest", response_model=IngestResponse)
async def ingest_api(body: IngestRequest, db: Session = Depends(get_db)):

    logger.info("🟢 /ingest called")
    logger.info(f"📥 Request → brand_id={body.brand_id}, classification_id={body.classification_id}, country={body.country}")

    # Basic validation
    if not body.brand_id:
        logger.error("❌ Missing brand_id")
        raise HTTPException(400, "brand_id is required")

    if not body.country:
        logger.error("❌ Missing country")
        raise HTTPException(400, "country is required")

    try:
        # REAL LOGIC lives in the service file
        result = await VeevaPlusLicense.run_ingestion(body,veeva,vectorstore,db)

        logger.info("✅ Ingestion completed successfully")
        return IngestResponse(inserted=result)

    except HTTPException as e:
        logger.error(f"❌ HTTP Error: {str(e.detail)}")
        raise

    except Exception as e:
        logger.error("❌ Unexpected ingestion error")
        logger.error(str(e))
        raise HTTPException(500, "Ingestion failed. Check logs.")
    

@router.post("/ingest-local", response_model=IngestResponse)
async def ingest_local_api(body: IngestRequest, db: AsyncSession = Depends(get_db)):
    """
    Router only performs:
    • minimal validation
    • logging
    • forwarding to service function
    """
    logger.info("🟢 /ingest-local called")
    logger.info(f"📥 created_date={body.created_date}")

    try:
        result = await VeevaPlusLicense.run_ingest_local(body, db)
        logger.info("✅ ingest-local completed successfully")
        return IngestResponse(inserted=result)

    except HTTPException as e:
        logger.error(f"❌ HTTP error: {e.detail}")
        raise

    except Exception as e:
        logger.error(f"❌ Unexpected error in /ingest-local: {str(e)}")
        raise HTTPException(500, "Local ingestion failed")
    

@router.post("/search", status_code=status.HTTP_200_OK)
async def search_api(
    body: dict,
    db: AsyncSession = Depends(get_db),
    limit: int = 20,
    offset: int = 0
):
    logger.info("API: search_api called")
    logger.debug(f"Request body keys: {list(body.keys()) if body else None}")
    logger.debug(f"limit={limit}, offset={offset}")

    try:
        # -----------------------------
        # VALIDATION
        # -----------------------------

        # -----------------------------
        # CALL SERVICE
        # -----------------------------
        result = await VeevaPlusLicense.veeva_plus_search(
            body=body,
            db=db,
            vectorstore=vectorstore,
            async_client=async_client,
            limit=limit,
            offset=offset
        )

        logger.info("API: search_api completed successfully")
        return result

    except HTTPException as e:
        logger.error(f"API HTTPException: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"API Exception: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, "Internal server error")


@router.get("/generate/{image_id}", status_code=status.HTTP_200_OK)
async def generate_license_metadata(image_id: str, db: AsyncSession = Depends(get_db)):
    logger.info("API: generate_license_metadata called")
    logger.debug(f"image_id = {image_id}")

    try:
        # -------------------------------------------
        # VALIDATION
        # -------------------------------------------
        if not image_id or not image_id.strip():
            logger.error("Validation failed: image_id missing")
            raise HTTPException(400, "image_id must not be empty")

        # -------------------------------------------
        # DELEGATE TO SERVICE
        # -------------------------------------------
        result = await VeevaPlusLicense.veeva_plus_generate(image_id=image_id,async_client=async_client, db=db)

        logger.info("API: generate_license_metadata completed successfully")
        return result

    except HTTPException as e:
        logger.error(f"API HTTPException: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"API Exception: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, "Internal server error")
    
    
@router.put("/edit/{image_id}", status_code=status.HTTP_200_OK)
async def edit_license_metadata(image_id: str, body: dict, db: AsyncSession = Depends(get_db)):
    logger.info("API: edit_license_metadata called")
    logger.debug(f"image_id = {image_id}")
    logger.debug(f"raw body keys = {list(body.keys()) if body else None}")

    try:
        # ----------------------------------------------------
        # BASIC VALIDATION
        # ----------------------------------------------------
        if not image_id or not image_id.strip():
            logger.error("Validation failed: image_id missing")
            raise HTTPException(400, "image_id is required")

        if not body:
            logger.error("Validation failed: request body missing")
            raise HTTPException(400, "Request body cannot be empty")

        # ----------------------------------------------------
        # CALL SERVICE LAYER
        # ----------------------------------------------------
        result = await VeevaPlusLicense.veeva_plus_edit(image_id=image_id, payload=body, db=db)

        logger.info("API: edit_license_metadata completed successfully")
        return result

    except HTTPException as e:
        logger.error(f"API HTTPException: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"API Exception: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, "Internal server error")




@router.post("/publish/{image_id}", status_code=status.HTTP_200_OK)
async def publish_to_veeva(image_id: str, db: AsyncSession = Depends(get_db)):
    logger.info("API: publish_to_veeva started")
    logger.debug(f"image_id = {image_id}")

    try:
        # -----------------------------------
        # BASIC VALIDATION
        # -----------------------------------
        if not image_id or len(image_id.strip()) == 0:
            logger.error("image_id empty")
            raise HTTPException(400, "image_id must not be empty")

        # -----------------------------------
        # CALL SERVICE LAYER
        # -----------------------------------
        result = await VeevaPlusLicense.veeva_plus_publish(image_id=image_id, db=db,vectorstore=vectorstore)

        logger.info("API: publish_to_veeva completed successfully")
        return result

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, "Internal server error")




@router.post("/license/{image_id}", status_code=status.HTTP_200_OK)
async def publish_to_veeva_license(image_id: str, db: AsyncSession = Depends(get_db)):
    logger.info("API: publish_to_veeva_license called")
    logger.debug(f"image_id = {image_id}")

    try:
        # ------------------------------
        # BASIC VALIDATION
        # ------------------------------
        if not image_id or len(image_id.strip()) == 0:
            logger.error("image_id is empty")
            raise HTTPException(400, "image_id must not be empty")

        # ------------------------------
        # CALL SERVICE LAYER
        # ------------------------------
        result = await VeevaPlusLicense.veeva_plus_license(image_id=image_id,async_client=async_client, db=db)

        logger.info("API: publish_to_veeva_license executed successfully")
        return result

    except HTTPException as e:
        logger.error(f"API HTTPException: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"API Exception: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, "Internal server error")
    




@router.post("/ingest-specific", response_model=IngestResponse)
async def ingest_api(body: IngestRequestSpecific, db: Session = Depends(get_db)):

    logger.info("🟢 /ingest called")
    logger.info(f"📥 Request → brand_id={body.brand_id}, classification_id={body.classification_id}, country={body.country}")

    # Basic validation
    if not body.brand_id:
        logger.error("❌ Missing brand_id")
        raise HTTPException(400, "brand_id is required")

    if not body.country:
        logger.error("❌ Missing country")
        raise HTTPException(400, "country is required")
    
    if not body.list:
        logger.error("❌ Missing List")
        raise HTTPException(400, "list is required")

    try:
        # REAL LOGIC lives in the service file
        result = await VeevaPlusLicense.run_ingestion_specific(body,veeva,vectorstore,db)

        logger.info("✅ Ingestion completed successfully")
        return IngestResponse(inserted=result)

    except HTTPException as e:
        logger.error(f"❌ HTTP Error: {str(e.detail)}")
        raise

    except Exception as e:
        logger.error("❌ Unexpected ingestion error")
        logger.error(str(e))
        raise HTTPException(500, "Ingestion failed. Check logs.")