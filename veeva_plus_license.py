from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from typing import Any, Dict, List, Tuple, Optional, Literal
import uuid
from jinja2 import Template
from xhtml2pdf import pisa
from io import BytesIO
import html as html_utils
import html as html_lib  # for escaping
from io import BytesIO
import asyncio
from app.schemas import (DownloadImageResponse, EnrichPrompt, FiltersResponse,CountriesResponse,GoToProjectResponse,
                         GenerateCompliantImage, GenerateImage,
                         GenerateImageResponse, ImageAnalysisResponse,
                         ImageParametersResponse,ProjectHistoryResponse, PromptSuggestions, ReferenceResponse, ReferenceUrl,GenerateMediaResponse,
                         ReferenceResponseList,TokenRequest,TokenResponse,IngestRequest,IngestResponse,ImageResult,SearchResult,SearchRequest,ProjectSearchRequest,ProjectResult,GenerateImageResponse, VariationResponse, SaveVariationsResponse, ImageAsset,ImageVersionResponse,ImageGroupDTO)
import json
from datetime import datetime, timezone
from datetime import date
from dateutil.relativedelta import relativedelta
from app.repository.utils import upload_to_s3, generate_presigned_url
from app.core.veeva import VeevaHelper
from reportlab.lib.pagesizes import letter
from sqlalchemy import select, func, and_,update,desc
from app.models import Country, Image, ImageType, Product, Project, Prompt,TherapeuticArea, VeevaDoc, LicenseFile,VeevaClassifications,VeevaBrand,VeevaAudience,VeevaCreativeAgency,VeevaSegment,VeevaUploader, VeevaTopics,PermittedChannels,PermittedCountries
from app.models.images import ImageType as Type
from app.repository import imagegen, utils
from app.repository.prompts import refine_prompt, restructure_prompt
from app.repository.utils import generate_presigned_url, generate_token,upload_to_s3
from app.schemas import (DownloadImageResponse, EditImage,EditProject, EnrichPrompt,ProjectListResponse,DeleteProjectResponse,
                         FiltersResponse, GenerateCompliantImage,
                         GenerateImage, GenerateImageResponse,
                         ImageAnalysisResponse, ImageParametersResponse,
                         PromptSuggestions, ReferenceResponse,EditProjectResponse,
                         ReferenceResponseList, ReferenceUrl,
                         SaveVariationsRequest, VariationRequest, ReuseComponentResponse)

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from io import BytesIO
import tempfile
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from app.core.veeva import VeevaHelper
from app.core.config import settings




embeddings = OpenAIEmbeddings(model=settings.EMBED_MODEL)


# ---------------------------------------------------------
def extract_active_values(field) -> str:
    """
    Extract values from JSONB field.

    Accepts:
        - None
        - {"id":..., "value":..., "is_active":...}
        - [{"id":..., "value":..., "is_active":...}, ...]

    Returns:
        Comma-separated string of ONLY records where is_active == True.
    """
    if not field:
        return ""

    # Single dict
    if isinstance(field, dict):
        return field.get("value", "") if field.get("is_active", False) else ""

    # List of dicts
    if isinstance(field, list):
        active_items = []
        for item in field:
            if isinstance(item, dict) and item.get("is_active", False):
                val = item.get("value", "")
                if val:
                    active_items.append(str(val))
        return ", ".join(active_items)

    # Fallback for unexpected shapes
    return str(field)

# def render_license_html(license_rec: LicenseFile) -> str:
#     esc = lambda s: html_utils.escape("" if s is None else str(s))

#     # Active JSON → string
#     audience = esc(extract_active_values(license_rec.audience))
#     permitted_countries = esc(extract_active_values(license_rec.permitted_countries))
#     permitted_channels = esc(extract_active_values(license_rec.permitted_channels))
#     indegene_logo_url=""

#     html = f"""
# <!DOCTYPE html>
# <html>
# <head>
# <meta charset="UTF-8" />
# <title>Indegene License Terms</title>

# <style>
#     @page {{
#         size: A4;
#         margin: 14mm;
#     }}

#     body {{
#         font-family: Helvetica, Arial, sans-serif;
#         font-size: 11pt;
#         background-color: white;
#         margin: 10mm;
#         padding: 0;
#     }}

#     .wrapper {{
#         width: 100%;
#     }}

#     .card {{
#         background: #ffffff;
#         border: 1px solid #d3d8e0;
#         padding: 26px 30px;
#         position: relative;
#     }}

#     .left-bar {{
#         position: absolute;
#         left: 0;
#         top: 0;
#         bottom: 0;
#         width: 8px;
#         background-color: #0054A6;
#     }}

#     .title {{
#         text-align:center;
#         margin-left: 1px;
#         font-size: 20pt;
#         font-weight: bold;
#         margin-bottom: 6px;
#         color: 'black';
#     }}

#     .section-title {{
#         color: #0054A6;
#         font-size: 14pt;
#         font-weight: bold;
#         margin-top: 22px;
#         margin-bottom: 6px;
#         border-bottom: 1px solid #0054A6;
#         padding-bottom: 3px;
#     }}

#     p {{
#         text-align: justify;
#         margin-top: 10px;
#         line-height: 1.45;
#     }}

#     .info-table {{
#         width: 100%;
#         border-collapse: collapse;
#         margin-top: 8px;
#     }}
#     .numbered-heading .underline {{
#          font-weight: bold;
#     }}

#     .info-table td {{
#         padding: 5px 2px;
#         vertical-align: top;
#     }}

#     td.label {{
#         width: 45%;
#         font-weight: bold;
#         color: #0054A6;
#         font-size: 10pt;
#     }}

#     td.value {{
#         width: 55%;
#         text-align: right;
#         font-size: 10pt;
#         color: #000000;
#         word-wrap: break-word;
#     }}

#     .hint {{
#         font-size: 8pt;
#         color: #777;
#     }}

#     .subpoint {{
#         margin-left: 18px;
#         font-size: 10.5pt;
#         margin-top: 6px;
#     }}

#     ul {{
#         margin-top: 4px;
#         margin-bottom: 8px;
#     }}

#     .annexure-box {{
#         margin-top: 28px;
#         border: 1px solid #0054A6;
#         padding: 18px 20px;
#         background: #f9fbff;
#     }}

#     .annexure-title {{
#         font-size: 14pt;
#         text-align: center;
#         color: #0054A6;
#         font-weight: bold;
#         margin-bottom: 12px;
#     }}
# </style>

# </head>
# <body>
# <div class="page"> <h1 class="title">Terms &amp; Conditions</h1> <p> The following are the terms &amp; conditions for usage of the Images generated for Gilead under the Work Order No. <strong>00694400.0</strong> dated <strong>1st July 2025</strong> and the platform provided under the Work Order Project Name: <strong>NEXT Licensing &amp; Languages</strong> dated <strong>December 15, 2023</strong> (collectively referred as the “Work Order”) between <strong>Gilead Sciences Europe Limited</strong> (“Gilead”) and <strong>Indegene, Inc.</strong> (“Indegene”). </p> <p> The details of specific Image generated for these terms &amp; conditions are provided in <strong>Annexure A</strong> attached herewith. </p> <p> The Images generated under the said Work Order will be published to Gilead’s content management platform <strong>Veeva PromoMats</strong> along with these terms &amp; conditions. The usage of the Images by Gilead shall be in compliance with these terms &amp; conditions. </p> <p class="numbered-heading"> 1. <span class="underline">Background:</span> </p> <p> These terms &amp; conditions set out the terms &amp; conditions for usage of Images generated via the Content Studio platform under the Work Order. Indegene has partnered with Adobe to leverage its Generative AI models for the creation of these components. The generated components (collectively referred as “Images”) are intended for use by Gilead, as the end user, in communication materials related to the mentioned brand’s product campaigns as per the Work Order. </p> <div class="footer-logo"> <img src="{esc(indegene_logo_url)}" alt="Indegene" /> </div> </div> <!-- ======================== PAGE 2 ======================== --> <div class="page"> <p class="numbered-heading"> 2. <span class="underline">License and Use of Images</span> </p> <p> <strong>i. License Grant:</strong> Indegene grants Gilead a non-exclusive, non-transferable, non-sublicensable license to use the Images, solely for the purpose of Gilead’s communication materials and campaigns as per the scope under the Work Order as follows: </p> <ol type="1"> <li> <strong>Communication Channels:</strong> Gilead may use the Images in Permitted Channels specified in Annexure A (currently: {permitted_channels}). It is hereby clarified that these Images shall not be used in any form of printed (Newspapers, magazines, brochures, etc.), social or electronic media (Radio, television, cinema, etc.). </li> <li> <strong>Territory:</strong> The license granted herein is for Permitted Countries specified in Annexure A (currently: {permitted_countries}). </li> <li> <strong>Audience:</strong> Gilead may use the Images for Audience specified in Annexure A (currently: {audience}). </li> <li> <strong>Duration of Use:</strong> The license shall commence from the License Start Date and continue till License Expiration Date as specified in Annexure A (currently: {esc(license_rec.license_start_date)} to {esc(license_rec.license_end_date)}). </li> <li> <strong>Termination of license:</strong> Notwithstanding anything mentioned in Annexure A, if the Work Order is terminated then the license shall automatically terminate, unless otherwise Parties mutually extend the Work Order. </li> <li> <strong>Image details:</strong> As per Annexure A. </li> </ol> <p> <strong>ii. Ownership and Rights:</strong> The image is generated using proprietary Gen AI technology owned by Indegene or GenAI technology licensed by Adobe to Indegene. Intellectual Property Rights (IP) in the Images and the underlying technology remain with the Indegene, including but not limited to any modifications, developments, changes and derivatives. The license grants Gilead the right to use the Image in accordance with the terms and conditions set forth herein. Gilead does not obtain any intellectual property rights in the Image. As between Indegene and Gilead, Indegene retains ownership of the Image. Indegene’s ownership in the Image, however, is subject to the applicable laws of the jurisdiction where the Image is used. </p> <p> <strong>iii. Restrictions on Use:</strong> The Images may not be used: </p> <ol type="a" class="subpoint"> <li> In any manner that is illegal or infringes the rights of any third party (including intellectual property, privacy, or publicity rights); </li> <li> To train, test, or improve any artificial intelligence or machine learning models; </li> <li> In any way that violates Adobe’s terms of use or the agreement between Indegene and Adobe. Reference: <ul> <li>Adobe Firefly Legal FAQs – Enterprise Customers</li> <li>PSLT – Adobe Creative Cloud, Firefly, and Substance 3D APIs (2024v3)</li> <li>Adobe General Terms (2024v1)</li> </ul> </li> <li> If Adobe materially changes its licensing terms &amp; conditions, Indegene reserves the right to update this License accordingly. </li> </ol> <div class="footer-logo"> <img src="{esc(indegene_logo_url)}" alt="Indegene" /> </div> </div> <!-- ======================== PAGE 3 ======================== --> <div class="page"> <p> <strong>iv. No Guarantee of Uniqueness:</strong> The Images generated by using Gen AI technology may not be unique or exclusive, and similar or identical Images may be generated by other users. </p> <p> <strong>v. Indemnification:</strong> Indegene will defend, at its expense, any third-party claim against Gilead made during the duration of use mentioned here. This indemnification is subject to usage of the Image in accordance with the terms and conditions mentioned herein and the limitations set forth by Adobe in the following links including clause vi below – </p> <ul> <li>Adobe Firefly Legal FAQs – Enterprise Customers</li> <li>PSLT – Adobe Creative Cloud, Firefly, and Substance 3D APIs (2024v3)</li> <li>Adobe General Terms (2024v1)</li> </ul> <ol type="a" class="subpoint"> <li> Indemnity will also not be applicable for Images generated using Adobe’s custom trained models. Gilead will be informed priorly through a written communication if these custom trained models are used to generate any of the Images in scope. </li> <li> NOTWITHSTANDING anything mentioned in the Work Order or Agreement Indegene and/or Adobe shall not be liable for any indirect, incidental, or consequential losses and the maximum aggregate liability of Indegene and/or Adobe shall be limited to the Fees payable under the applicable Work Order during the 12 months immediately preceding the claim giving rise to such liability. </li> <li> Indegene cannot be held responsible for any claims arising out of Gilead’s use of the Images in violation of this Agreement or applicable law. </li> </ol> <p> <strong>vi. Limitation of Liability:</strong> Indegene or Adobe shall not be liable for any claims arising from: </p> <ol type="a" class="subpoint"> <li>Modifications to the Images by Gilead or third parties;</li> <li>Combination of the Images with other material, content or information;</li> <li>Use of the Images in violation of this agreement;</li> <li>The context in which the Images are used;</li> <li>Continued use of the Images after notice to cease such use.</li> </ol> <p> <strong>vii. Compliance:</strong> The responsibility on usage of the Images lies with Gilead. Gilead is solely responsible for ensuring that its use of the Images complies with applicable laws, regulations, and industry codes, including pharmaceutical advertising standards. Gilead has to ensure it complies with all applicable laws and regulations, as well as terms and conditions mentioned herein, in its use of the Images. </p> <p> <strong>viii. Rendition Generation by Veeva:</strong> Once the Image is published from Content Studio to Gilead’s content management platform <strong>Veeva PromoMats</strong>, the uploaded content is automatically rendered in different formats and sizes by Veeva Vault PromoMats. Indegene has no role in this activity and shall not be held liable for any infringement claims on the generated renditions. </p> <p> <strong>This document is generated by Indegene to clarify the terms &amp; conditions to be followed by Gilead for usage of the Images generated by Indegene under the Work Order.</strong> </p> <div class="footer-logo"> <img src="{esc(indegene_logo_url)}" alt="Indegene" /> </div> </div> <!-- ======================== PAGE 4 – ANNEXURE A ======================== --> <div class="page"> <h2 class="annexure-title">Annexure A</h2> <table class="annexure"> <tr> <th class="section-header" colspan="2">Component Information</th> </tr> <tr> <td class="label">Name</td> <td class="value"> {esc(license_rec.name)} <span style="font-size: 8pt;"> (max. 100 characters)</span> </td> </tr> <tr> <td class="label">Title</td> <td class="value">{esc(license_rec.title)}</td> </tr> <tr> <td class="label">Country</td> <td class="value">{esc(license_rec.country)}</td> </tr> <tr> <td class="label">Product</td> <td class="value">{esc(license_rec.product)}</td> </tr> <tr> <td class="label">Audience</td> <td class="value">{audience}</td> </tr> <tr> <td class="label">Content Studio Image ID</td> <td class="value">{esc(license_rec.image_id)}</td> </tr> <tr> <th class="section-header" colspan="2">Rights and Licensing Information</th> </tr> <tr> <td class="label">License Start Date</td> <td class="value">{esc(license_rec.license_start_date)}</td> </tr> <tr> <td class="label">License Expiration Date</td> <td class="value">{esc(license_rec.license_end_date)}</td> </tr> <tr> <td class="label">Permitted Countries</td> <td class="value">{permitted_countries}</td> </tr> <tr> <td class="label">Permitted Channels</td> <td class="value">{permitted_channels}</td> </tr> </table> <div class="footer-logo"> <img src="{esc(indegene_logo_url)}" alt="Indegene" /> </div> </div> </body> </html>
# """
#     return html


def render_license_html(license_rec: LicenseFile) -> str:
    esc = lambda s: html_utils.escape("" if s is None else str(s))

    audience = esc(extract_active_values(license_rec.audience))
    permitted_countries = esc(extract_active_values(license_rec.permitted_countries))
    permitted_channels = esc(extract_active_values(license_rec.permitted_channels))
    indegene_logo_url = ""  # add your logo URL or base64

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Terms & Conditions</title>

<style>

    /* RESET */
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    @page {{
        size: A4;
        margin: 18mm 18mm 18mm 18mm;
    }}

    body {{
        font-family: Arial, Helvetica, sans-serif;
        font-size: 11pt;
        line-height: 1.20;
        background: #fff;
    }}

    .page {{
        width: 100%;
        min-height: 260mm;
        padding: 6mm 5mm 6mm 7mm;
        position: relative;
        page-break-after: always;
    }}

    .footer-logo {{
        position: absolute;
        right: 3mm;
        bottom: 8mm;
    }}

    .footer-logo img {{
        max-height: 18mm;
    }}

    p {{
        margin: 3pt 0;
        padding-left: 3mm;
        text-align: justify;
        line-height: 1.20;
    }}

    h1 {{
        font-size: 13.5pt;
        font-weight: bold;
        margin-bottom: 8pt;
        padding-left: 3mm;
    }}

    .clause-title {{
        font-weight: bold;
        margin: 6pt 0 3pt 0;
        padding-left: 3mm;
    }}

    .sub-title {{
        font-weight: bold;
        margin: 5pt 0 3pt 0;
        padding-left: 3mm;
    }}

    ol, ul {{
        margin: 3pt 0 3pt 18pt;
        padding: 0;
    }}

    li {{
        margin: 2pt 0;
        line-height: 1.20;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 6mm;
        font-size: 11pt;
    }}

    td, th {{
        border: 1px solid #000;
        padding: 4pt 5pt;
        vertical-align: top;
    }}

    th {{
        background: #f2f2f2;
        font-weight: bold;
    }}

</style>
</head>
<body>


<!-- PAGE 1 -->
<div class="page">

<h1>Terms &amp; Conditions</h1>

<p>The following are the terms &amp; conditions for usage of the Images generated for Gilead under
the Work Order No. 00694400.0 dated 1st July 2025 and the platform provided under the Work
Order Project Name: NEXT Licensing &amp; Languages dated December 15, 2023 (collectively
referred as “Work Order”) between Gilead Sciences Europe Limited (“Gilead”) and Indegene,
Inc. (“Indegene”).</p>

<p>The details of specific Image generated for these terms &amp; conditions are provided in Annexure
A attached herewith.</p>

<p>The Images generated under the said Work Order will be published to Gilead’s content
management platform “Veeva PromoMats” along with these terms &amp; conditions. The usage of
the Images by Gilead shall be in compliance with these terms &amp; conditions.</p>


<p class="clause-title">1. Background:</p>

<p>These terms &amp; conditions set out the terms &amp; conditions for usage of Images generated via
the Content Studio platform under the Work Order. Indegene has partnered with Adobe to
leverage its Generative AI models for the creation of these components. The generated
components (collectively referred as “Images”) are intended for use by Gilead, as the end
user, in communication materials related to the mentioned brand’s product campaigns as per
the Work Order.</p>


<p class="clause-title">2. License and Use of Images</p>

<p class="sub-title">i. License Grant:</p>

<ol>
<li>Communication Channels: Gilead may use the Images in Permitted Channels specified in
Annexure A. It is hereby clarified that these Images shall not be used in any form of printed
(Newspapers, magazines, brochures, etc.), social or electronic media (Radio, television,
cinema, etc.).</li>

<li>Territory: The license granted herein is for Permitted Countries specified in Annexure A.</li>

<li>Audience: Gilead may use the Images for Audience specified in Annexure A.</li>

<li>Duration of Use: The license shall commence from the License Start Date and continue till
License Expiration Date as specified in Annexure A.</li>
</ol>

<!-- FORCE PAGE BREAK HERE -->
</div>



<!-- PAGE 2 -->
<div class="page">

<p class="sub-title">ii. License Grant (Continued):</p>

<ol start="5">
<li>Termination of license: Notwithstanding anything mentioned in Annexure A if the Work Order
is terminated then the license shall automatically terminate, unless otherwise Parties mutually
extend the Work Order.</li>

<li>Image details: As per Annexure A.</li>
</ol>


<p class="sub-title">iii. Ownership and Rights:</p>

<p>The image is generated using proprietary Gen AI technology owned by Indegene or GenAI
technology licensed by Adobe to Indegene. Intellectual Property Rights (IP) in the Images and
the underlying technology remain with the Indegene, including but not limited to any
modifications, developments, changes and derivatives. The license grants Gilead the right to
use the Image in accordance with the terms and conditions set forth herein. Gilead does not
obtain any intellectual property rights in the Image. As between Indegene and Gilead,
Indegene retains ownership of the Image.</p>

<p class="sub-title">iv. Restrictions on Use:</p>

<ol type="a">
<li>Use in illegal or infringing manner;</li>
<li>Use for AI training or ML model improvement;</li>
<li>Use violating Adobe policies.</li>
</ol>

<p class="sub-title">v. No Guarantee of Uniqueness:</p>

<p>Images may not be unique or exclusive.</p>


<p class="sub-title">vi. Indemnification:</p>

<p>Indegene will defend Gilead for valid claims subject to Adobe limitations.</p>


<p class="footer-logo">
    <img src="{indegene_logo_url}">
</p>

</div>



<!-- PAGE 3 — ANNEXURE A -->
<div class="page">

<h1 style="font-size:13pt; text-align:center; padding-left:0;">Annexure A</h1>

<table>
<tr><th colspan="2">Component Information</th></tr>

<tr><td><b>Name</b></td><td>{esc(license_rec.name)}</td></tr>
<tr><td><b>Title</b></td><td>{esc(license_rec.title)}</td></tr>
<tr><td><b>Country</b></td><td>{permitted_countries}</td></tr>
<tr><td><b>Product</b></td><td>{esc(license_rec.product)}</td></tr>
<tr><td><b>Audience</b></td><td>{audience}</td></tr>

<tr><th colspan="2">Rights &amp; Licensing</th></tr>

<tr><td><b>License Start Date</b></td><td>{esc(license_rec.license_start_date)}</td></tr>
<tr><td><b>License End Date</b></td><td>{esc(license_rec.license_end_date)}</td></tr>
<tr><td><b>Permitted Channels</b></td><td>{permitted_channels}</td></tr>
<tr><td><b>Permitted Countries</b></td><td>{permitted_countries}</td></tr>

</table>

<p class="footer-logo">
    <img src="{indegene_logo_url}">
</p>

</div>

</body>
</html>
"""
    return html



# ---------------------------------------------------------
# Core PDF generator
# ---------------------------------------------------------

def generate_license_pdf(license_rec: LicenseFile) -> bytes:
    """
    Generate License PDF using xhtml2pdf (pure Python).

    Returns:
        pdf_bytes (bytes): Content of the generated PDF.
    """
    html = render_license_html(license_rec)

    pdf_buffer = BytesIO()
    pdf = pisa.CreatePDF(html, dest=pdf_buffer)

    if pdf.err:
        # Optionally log html or error details here
        raise Exception("PDF generation failed (xhtml2pdf error)")

    return pdf_buffer.getvalue()



def _parse_date(value: Any) -> Optional[datetime]:
    """Normalize incoming date (str/datetime/None) to datetime or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # epoch seconds (just in case)
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        # try ISO first; fall back to common formats if needed
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z",
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d"):
            try:
                dt = datetime.strptime(value, fmt)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
        # last resort
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None
    return None



def proper_case(value: str) -> str:
    if not value:
        return ""
    return value[0].upper() + value[1:]



class VeevaPlusLicense:

    @staticmethod
    async def veeva_plus_license(image_id: str, db: AsyncSession):
        logger.info("Service: veeva_plus_license started")

        # ---------------------------------------------------
        # STEP 1 — FETCH LICENSE FILE
        # ---------------------------------------------------
        logger.info("Step 1: Fetching LicenseFile")
        result = await db.execute(
            select(LicenseFile).where(LicenseFile.image_id == image_id)
        )
        license_rec = result.scalar_one_or_none()

        if not license_rec:
            logger.error("LicenseFile not found")
            raise HTTPException(404, "LicenseFile not found")

        logger.debug(f"LicenseFile.id = {license_rec.id}")
        logger.debug(f"title = {license_rec.title}")
        logger.debug(f"name = {license_rec.name}")
        logger.debug(f"product = {license_rec.product}")

        # ---------------------------------------------------
        # STEP 1B — REQUIRED FIELD VALIDATION
        # ---------------------------------------------------
        logger.info("Step 1B: Validating required fields")
        required_fields = [
            "title",
            "name",
            "product",
            "topic",
            "license_start_date",
            "license_end_date"
        ]

        for field in required_fields:
            value = getattr(license_rec, field, None)
            logger.debug(f"{field} = {value}")
            if not value:
                logger.error(f"Missing required field: {field}")
                raise HTTPException(400, f"Missing required field: {field}")

        # ---------------------------------------------------
        # STEP 2 — GENERATE PDF
        # ---------------------------------------------------
        logger.info("Step 2: Generating License PDF")
        pdf_bytes = generate_license_pdf(license_rec)
        logger.debug(f"PDF size bytes = {len(pdf_bytes)}")

        # ---------------------------------------------------
        # STEP 3 — UPLOAD TO S3
        # ---------------------------------------------------
        logger.info("Step 3: Uploading PDF to S3")
        pdf_key = f"license_pdf/{license_rec.id}.pdf"
        logger.debug(f"pdf_key = {pdf_key}")

        pdf_stream = BytesIO(pdf_bytes)
        pdf_s3_url = upload_to_s3(
            file=pdf_stream,
            file_key=pdf_key,
            content_type="application/pdf"
        )

        if not pdf_s3_url:
            logger.error("PDF upload to S3 failed")
            raise HTTPException(500, "Failed to upload PDF to S3")

        logger.debug(f"s3_url = {pdf_s3_url}")

        # ---------------------------------------------------
        # STEP 4 — BRAND LOOKUP
        # ---------------------------------------------------
        logger.info("Step 4: Getting Product Brand ID")
        brand_result = await db.execute(
            select(VeevaBrand).where(
                func.lower(VeevaBrand.value) == func.lower(license_rec.product)
            )
        )
        brand = brand_result.scalar_one_or_none()

        if not brand:
            logger.error("Brand not found in DB")
            raise HTTPException(404, f"No VeevaBrand found for {license_rec.product}")

        logger.debug(f"brand.veeva_id = {brand.veeva_id}")

        # ---------------------------------------------------
        # STEP 5 — TOPIC ID
        # ---------------------------------------------------
        logger.info("Step 5: Extracting topic_id")
        topic_id = (
            license_rec.topic[0].get("veeva_id")
            if isinstance(license_rec.topic, list)
            and license_rec.topic
            else None
        )

        if not topic_id:
            logger.error("topic_id missing in topic field")
            raise HTTPException(400, "topic_id missing")

        logger.debug(f"topic_id = {topic_id}")

        # ---------------------------------------------------
        # STEP 6 — UPLOAD TO VEEVA
        # ---------------------------------------------------
        logger.info("Step 6: Uploading document to Veeva")
        veeva = VeevaHelper()

        session_token = veeva.veeva_auth()
        logger.debug("Veeva session obtained")

        presigned_url = generate_presigned_url(pdf_key)
        logger.debug(f"presigned_url = {presigned_url}")

        veeva_res = veeva.upload_document_license(
            session_id=session_token,
            rights_other__v=license_rec.rights_other__v,
            s3_url=presigned_url,
            tags=license_rec.tags,
            name=license_rec.name,
            title=license_rec.title,
            product_id=brand.veeva_id,
            therapeutic_area=license_rec.therapeutic_area,
            topic_id=topic_id,
            is_rights_managed=True,
            license_start_date=license_rec.license_start_date,
            license_end_date=license_rec.license_end_date,
            permitted_countries=license_rec.permitted_countries,
            audience=license_rec.audience,
            segment=license_rec.segment,
            uploader_role=license_rec.uploader_role,
            creative_agency=license_rec.creative_agency,
            permitted_channels=license_rec.permitted_channels,
            country=license_rec.country,
            topic=license_rec.topic,
        )

        logger.debug(f"veeva_response = {veeva_res}")

        veeva_doc_id = (
            veeva_res.get("id")
            or veeva_res.get("document_id")
            or veeva_res.get("data", {}).get("document", {}).get("id")
        )

        if not veeva_doc_id:
            logger.error("document_id missing in Veeva response")
            raise HTTPException(500, "document_id missing in Veeva response")

        logger.debug(f"veeva_doc_id = {veeva_doc_id}")

        # ---------------------------------------------------
        # STEP 7 — UPDATE DB
        # ---------------------------------------------------
        logger.info("Step 7: Updating DB record")

        license_rec.is_license_generated = True
        license_rec.license_doc = str(veeva_doc_id)

        await db.commit()
        logger.info("DB commit successful")

        return {
            "status": "success",
            "veeva_id": veeva_doc_id,
            "pdf_used_for_upload": pdf_s3_url,
            "veeva_name":license_rec.name
        }

    async def veeva_plus_publish(image_id: str, db: AsyncSession, vectorstore):
        logger.info("Service: veeva_plus_publish started")

        logger.info("Service: veeva_plus_generate started")
        logger.debug(f"image_id = {image_id}")

        # ---------------------------------------------------
        # STEP 1 — FETCH IMAGE
        # ---------------------------------------------------
        logger.info("Step 1: Fetching Image record")

        res = await db.execute(select(Image).where(Image.id == image_id))
        image = res.scalar_one_or_none()

        if not image:
            logger.error("Image not found")
            raise HTTPException(404, f"No image found for {image_id}")

        # ---------------------------------------------------
        # STEP 2 — FETCH PROJECT
        # ---------------------------------------------------
        logger.info("Step 2: Fetching related Project")

        res = await db.execute(select(Project).where(Project.id == image.project_id))
        project = res.scalar_one_or_none()

        if not project:
            logger.error("Project not found")
            raise HTTPException(404, f"No project found for {image.project_id}")

        classification = project.type or "UnknownClassification"


        # ---------------------------------------------------
        # STEP 1 — FETCH LICENSE FILE
        # ---------------------------------------------------
        logger.info("Step 1: Fetch LicenseFile")
        result = await db.execute(
            select(LicenseFile).where(LicenseFile.image_id == image_id)
        )
        license_rec = result.scalar_one_or_none()

        if not license_rec:
            logger.error("LicenseFile not found")
            raise HTTPException(404, f"No LicenseFile for {image_id}")

        logger.debug(f"LicenseFile.id = {license_rec.id}")

        # ---------------------------------------------------
        # STEP 1B — REQUIRED VALIDATION
        # ---------------------------------------------------
        logger.info("Step 1B: Validating required fields")

        required = ["title", "name", "product", "topic", "s3_url"]
        for f in required:
            val = getattr(license_rec, f, None)
            logger.debug(f"{f} = {val}")
            if not val:
                logger.error(f"Missing required field: {f}")
                raise HTTPException(400, f"Missing required field: {f}")

        # ---------------------------------------------------
        # STEP 2 — BRAND LOOKUP
        # ---------------------------------------------------
        logger.info("Step 2: Brand lookup")

        brand_result = await db.execute(
            select(VeevaBrand).where(
                func.lower(VeevaBrand.value) == func.lower(license_rec.product)
            )
        )
        brand = brand_result.scalar_one_or_none()

        if not brand:
            logger.error("Brand not found")
            raise HTTPException(404, f"No brand for {license_rec.product}")

        brand_value = brand.value
        product_vid = brand.veeva_id

        # ---------------------------------------------------
        # STEP 3 — TOPIC ID
        # ---------------------------------------------------
        logger.info("Step 3: Extract topic_id")

        topic_id = (
            license_rec.topic[0].get("veeva_id")
            if isinstance(license_rec.topic, list) and license_rec.topic
            else None
        )

        if not topic_id:
            logger.error("topic_id missing")
            raise HTTPException(400, "topic_id missing in LicenseFile.topic")

        # ---------------------------------------------------
        # STEP 4 — UPLOAD TO VEEVA
        # ---------------------------------------------------
        logger.info("Step 4: Upload document to Veeva")

        v = VeevaHelper()
        token_query = v.auth()
        token_upload = v.veeva_auth()

        presigned = generate_presigned_url(license_rec.s3_url)

        upload_res = v.upload_document(
            session_id=token_upload,
            rights_other__v=license_rec.rights_other__v,
            s3_url=presigned,
            tags=license_rec.tags,
            name=license_rec.name,
            title=license_rec.title,
            product_id=product_vid,
            therapeutic_area=license_rec.therapeutic_area,
            topic_id=topic_id,
            classification = proper_case(classification),
            is_rights_managed=True,
            license_start_date=license_rec.license_start_date,
            license_end_date=license_rec.license_end_date,
            permitted_countries=license_rec.permitted_countries,
            audience=license_rec.audience,
            segment=license_rec.segment,
            uploader_role=license_rec.uploader_role,
            creative_agency=license_rec.creative_agency,
            permitted_channels=license_rec.permitted_channels,
            country=license_rec.country,
            topic=license_rec.topic,
            is_license=license_rec.is_license_generated,
            license_doc=license_rec.license_doc
        )

        vid = (
            upload_res.get("data", {}).get("document", {}).get("id")
            or upload_res.get("document_id")
            or upload_res.get("id")
        )

        if not vid:
            logger.error("No veeva_id returned")
            raise HTTPException(500, "Veeva upload succeeded but no ID returned")

        license_rec.veeva_id = str(vid)
        await db.commit()

        # ---------------------------------------------------
        # STEP 5 — UPDATE IMAGE
        # ---------------------------------------------------
        logger.info("Step 5: Update Image row")
        

        # ---------------------------------------------------
        # STEP 6 — FETCH METADATA FROM VQL
        # ---------------------------------------------------
        logger.info("Step 6: Fetch metadata via VQL")

        vql = f"""
        SELECT id, name__v, title__v, description__v, document_number__v,
               classification__v, country__v, status__v,
               license_start_date__c, license_expiration_date__c,
               is_rights_managed__c, audience_code__c, segment__c,
               uploader_role__c, creative_agency1__c, tags__v,
               version_id, version_modified_date__v
        FROM documents WHERE id = '{vid}'
        """

        rows = v.query_vql(token_query, vql)

        if not rows:
            logger.error("No metadata returned from Veeva")
            raise HTTPException(404, "No metadata from Veeva for this ID")

        r = rows[0]

        # ---------------------------------------------------
        # STEP 7 — NORMALIZATION
        # ---------------------------------------------------
        logger.info("Step 7: Normalizing fields")

        def normalize_field(val):
            if val is None:
                return []
            if isinstance(val, list):
                return [str(v).strip() for v in val if v]
            if isinstance(val, str):
                s = val.strip()
                if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                    try:
                        obj = json.loads(s)
                        if isinstance(obj, list):
                            return [str(v).strip() for v in obj if v]
                        return [str(obj).strip()]
                    except Exception:
                        if "," in s:
                            return [v.strip() for v in s.split(",") if v.strip()]
                        return [s]
                return [s]
            return [str(val).strip()]

        def to_str(v):
            if isinstance(v, list):
                return ", ".join(v) if len(v) > 1 else (v[0] if v else "")
            return str(v).strip() if v else ""

        # extract normalized values
        title = (r.get("title__v") or "").strip()
        name = (r.get("name__v") or "").strip()
        snippet = (r.get("description__v") or "").strip()
        docnum = (r.get("document_number__v") or "").strip()
        classification = r.get("classification__v")
        country = normalize_field(r.get("country__v"))
        status = r.get("status__v")
        lstart = r.get("license_start_date__c")
        lend = r.get("license_expiration_date__c")
        rights = r.get("is_rights_managed__c")
        audience = normalize_field(r.get("audience_code__c"))
        segment = normalize_field(r.get("segment__c"))
        uploader = normalize_field(r.get("uploader_role__c"))
        agency = normalize_field(r.get("creative_agency1__c"))
        tags = normalize_field(r.get("tags__v"))
        verid = r.get("version_id")
        verdate = r.get("version_modified_date__v")

        geography_str = to_str(country)

        # ---------------------------------------------------
        # STEP 8 — UPSERT INTO VEEVA_DOCS
        # ---------------------------------------------------
        logger.info("Step 8: Upsert into VeevaDoc")

        doc_q = await db.execute(
            select(VeevaDoc).where(VeevaDoc.veeva_id == str(vid))
        )
        doc = doc_q.scalars().first()

        # reembed = False

        # if not doc:
        #     doc = VeevaDoc(
        #         veeva_id=str(vid),
        #         brand=brand_value,
        #         geography=geography_str,
        #         classification = classification,
        #         name=name,
        #         title=title,
        #         snippet=snippet,
        #         document_number=docnum,
        #         status=status,
        #         license_start_date=lstart,
        #         license_expiration_date=lend,
        #         is_rights_managed=rights,
        #         country=country,
        #         audience_code=audience,
        #         segment=segment,
        #         uploader_role=uploader,
        #         creative_agency=agency,
        #         tags=tags,
        #         version_id=verid,
        #         version_date=verdate,
        #         s3_url=license_rec.s3_url,
        #         rights_other__v=license_rec.rights_other__v
        #     )
        #     db.add(doc)
        #     reembed = True
        # else:
        #     doc.title = title
        #     doc.name = name
        #     doc.snippet = snippet
        #     doc.classification = classification
        #     doc.geography = geography_str
        #     doc.status = status
        #     doc.document_number = docnum
        #     doc.license_start_date = lstart
        #     doc.license_expiration_date = lend
        #     doc.is_rights_managed = rights
        #     doc.country = country
        #     doc.audience_code = audience
        #     doc.segment = segment
        #     doc.uploader_role = uploader
        #     doc.creative_agency = agency
        #     doc.tags = tags
        #     doc.version_id = verid
        #     doc.version_date = verdate
        #     doc.s3_url = license_rec.s3_url
        #     doc.rights_other__v = license_rec.rights_other__v
        #     reembed = True

        # await db.commit()

        # # ---------------------------------------------------
        # # STEP 9 — EMBEDDING
        # # ---------------------------------------------------
        # logger.info("Step 9: PGVector embedding")

        # if reembed:
        #     text = f"Title: {title} | Name: {name} | Doc#: {docnum}\n\n{snippet}"

        #     metadata = {
        #         "id": str(vid),
        #         "title": title,
        #         "name": name,
        #         "snippet": snippet,
        #         "document_number": docnum,
        #         "classification": classification,
        #         "brand": brand_value.lower(),
        #         "geography": geography_str.lower(),
        #         "status": status,
        #         "license_start_date": lstart,
        #         "license_expiration_date": lend,
        #         "is_rights_managed": rights,
        #         "audience_code": audience,
        #         "segment": segment,
        #         "uploader_role": uploader,
        #         "creative_agency": agency,
        #         "tags": tags,
        #         "version_id": verid,
        #         "version_date": verdate,
        #         "s3_url": license_rec.s3_url,
        #     }

        #     vector_id = f"veeva::{vid}"

        #     try:
        #         vectorstore.delete(ids=[vector_id])
        #     except Exception:
        #         pass

        #     vectorstore.add_texts(
        #         texts=[text],
        #         metadatas=[metadata],
        #         ids=[vector_id]
        #     )



        img_result = await db.execute(select(Image).where(Image.id == image_id))
        img = img_result.scalar_one_or_none()

        if img:
            img.is_published = True
            img.veeva_published_number=str(docnum)
            img.veeva_published_status='Draft'
            await db.commit()
        # ---------------------------------------------------
        # FINAL RESPONSE
        # ---------------------------------------------------
        return {
            "status": "success",
            "message": f"{title} published successfully.",
            "veeva_id": str(vid),
            "veeva_doc_number": docnum,
        }
    

    async def veeva_plus_edit(image_id: str, payload: dict, db: AsyncSession):
        logger.info("Service: veeva_plus_edit started")
        logger.debug(f"image_id = {image_id}")

        # ---------------------------------------------------
        # STEP 1 — FETCH RECORD
        # ---------------------------------------------------
        logger.info("Step 1: Fetching LicenseFile record")

        result = await db.execute(
            select(LicenseFile).where(LicenseFile.image_id == image_id)
        )
        license_record = result.scalar_one_or_none()

        if not license_record:
            logger.error("LicenseFile not found for given image_id")
            raise HTTPException(404, f"No license record found for {image_id}")

        logger.debug(f"license_record.id = {license_record.id}")

        # ---------------------------------------------------
        # STEP 2 — EXTRACT EDITABLE FIELDS
        # ---------------------------------------------------
        logger.info("Step 2: Extracting editable fields")

        updates = {
            "title": payload.get("title"),
            "subtitle": payload.get("subtitle"),
            "name": payload.get("name"),

            "audience": payload.get("audience_options"),
            "segment": payload.get("segment_options"),
            "creative_agency": payload.get("creative_agency_options"),
            "uploader_role": payload.get("uploader_role_options"),
            "permitted_channels": payload.get("permitted_channel_options"),
            "permitted_countries": payload.get("permitted_country_options"),
            "topic": payload.get("topic_options"),
            "descriptor":payload.get("descriptor"),
            "license_start_date": payload.get("default_fields", {}).get("license_start_date"),
            "license_end_date": payload.get("default_fields", {}).get("license_end_date"),
            "rights_other__v": payload.get("default_fields", {}).get("rights_other__v"),

            # Reset license flags on edit
            "is_license_generated": False,
            "license_doc": ""
        }

        # Log extracted field summary
        for k, v in updates.items():
            logger.debug(f"{k} = {v}")

        # ---------------------------------------------------
        # STEP 3 — FILTER OUT UNCHANGED (None) FIELDS
        # ---------------------------------------------------
        logger.info("Step 3: Filtering unchanged fields")

        updates = {k: v for k, v in updates.items() if v is not None}
        logger.debug(f"Fields to update = {list(updates.keys())}")

        if not updates:
            logger.warning("No fields to update")
            raise HTTPException(400, "No fields provided for update")

        # ---------------------------------------------------
        # STEP 4 — APPLY UPDATES
        # ---------------------------------------------------
        logger.info("Step 4: Applying DB updates")

        await db.execute(
            update(LicenseFile).where(LicenseFile.image_id == image_id).values(**updates)
        )
        await db.commit()

        logger.info("DB update committed successfully")

        # ---------------------------------------------------
        # STEP 5 — FETCH UPDATED RECORD
        # ---------------------------------------------------
        logger.info("Step 5: Fetching updated record")

        refreshed = await db.execute(
            select(LicenseFile).where(LicenseFile.image_id == image_id)
        )
        updated_record = refreshed.scalar_one()

        # ---------------------------------------------------
        # STEP 6 — BUILD RESPONSE JSON (SAME STRUCTURE AS FRONTEND EXPECTS)
        # ---------------------------------------------------
        logger.info("Step 6: Building response JSON")

        response = {
            "image_id": image_id,
            "image_name": updated_record.name,
            "image_number":updated_record.image_number,
            "product": updated_record.product,
            "country": updated_record.country,
            "classification": payload.get("classification"),
            "therapeutic_area": updated_record.therapeutic_area,
            "language": updated_record.language,
            "resolution": payload.get("resolution"),
            "tags": updated_record.tags,
            "format": updated_record.format,
            "name": updated_record.name,
            "subtitle": updated_record.subtitle,
            "s3_url": updated_record.s3_url,
            "title": updated_record.title,

            # options returned as stored
            "topic_options": updated_record.topic,
            "audience_options": updated_record.audience,
            "segment_options": updated_record.segment,
            "uploader_role_options": updated_record.uploader_role,
            "creative_agency_options": updated_record.creative_agency,
            "permitted_channel_options": updated_record.permitted_channels,
            "permitted_country_options": updated_record.permitted_countries,

            "is_license_generated": updated_record.is_license_generated,
            "license_doc": updated_record.license_doc,
            "descriptor":updated_record.descriptor,
            "default_fields": {
                "license_start_date": updated_record.license_start_date,
                "license_end_date": updated_record.license_end_date,
                "rights": "All Generated Image",
                "publish": True,
                "rights_other__v": updated_record.rights_other__v
            }
        }

        logger.info("Service: veeva_plus_edit finished successfully")
        return response
    

    async def veeva_plus_generate(image_id: str,async_client, db: AsyncSession):
        logger.info("Service: veeva_plus_generate started")
        logger.debug(f"image_id = {image_id}")

        # ---------------------------------------------------
        # STEP 1 — FETCH IMAGE
        # ---------------------------------------------------
        logger.info("Step 1: Fetching Image record")

        res = await db.execute(select(Image).where(Image.id == image_id))
        image = res.scalar_one_or_none()

        if not image:
            logger.error("Image not found")
            raise HTTPException(404, f"No image found for {image_id}")

        # ---------------------------------------------------
        # STEP 2 — FETCH PROJECT
        # ---------------------------------------------------
        logger.info("Step 2: Fetching related Project")

        res = await db.execute(select(Project).where(Project.id == image.project_id))
        project = res.scalar_one_or_none()

        if not project:
            logger.error("Project not found")
            raise HTTPException(404, f"No project found for {image.project_id}")

        brand = project.product or "UnknownBrand"
        country = project.country or "UnknownCountry"
        classification = project.type or "UnknownClassification"

        # ---------------------------------------------------
        # STEP 3 — FETCH PROMPT
        # ---------------------------------------------------
        logger.info("Step 3: Fetching prompt/enhanced_prompt")

        res = await db.execute(
            select(Prompt.prompt, Prompt.enhanced_prompt).where(Prompt.id == image.prompt_id)
        )
        row = res.first()

        if not row:
            logger.error("Prompt not found")
            raise HTTPException(404, "Prompt not found")

        prompt_title, enhanced_prompt = row

        title = prompt_title or f"{brand} visual creative asset"
        subtitle = f"Generated automatically for {brand} in {country}"

        # ---------------------------------------------------
        # STEP 4 — GENERATE TAGS
        # ---------------------------------------------------
        logger.info("Step 4: Generating tags")

        tags = []
        if enhanced_prompt:
            tag_prompt = f"""
            Generate 5–10 short, meaningful tags (1–2 words).
            Return ONLY a JSON list of lowercase strings.
            Concept: "{enhanced_prompt}"
            """

            resp = await async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": tag_prompt}],
                temperature=0.4
            )

            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()

            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    tags = [t.strip().lower() for t in parsed]
            except Exception:
                tags = [t.strip().lower() for t in raw.split(",") if t.strip()]

            blacklist = {"image", "photo", "picture", "art", "graphic"}
            tags = [t for t in tags if t not in blacklist]

        descriptor_string=""


        if enhanced_prompt:
            descriptor_prompt = f"""
            Generate 50 character short, meaningful title.
            Return ONLY a string.
            Concept: "{enhanced_prompt}"
            """

            resp = await async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": descriptor_prompt}],
                temperature=0.4
            )

            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()

            try:
                parsed = json.loads(raw)

                descriptor_string=parsed

            except Exception:
                descriptor_string=""

        # ---------------------------------------------------
        # STEP 5 — GENERATE NAME
        # ---------------------------------------------------
        logger.info("Step 5: Generating name")

        img_name = str(image.image_name)
        if "_" in img_name:
            suffix = img_name.split("_")[-1].lstrip("0") or img_name[-3:]
        else:
            suffix = img_name[-4:]

        name = f"{proper_case(brand)}_{proper_case(classification)}_{descriptor_string}_Indegene_{image.image_name}_{date(date.today().year, 12, 31).strftime("%Y-%m-%d")}"

        # ---------------------------------------------------
        # STEP 6 — FETCH ALL DROPDOWNS
        # ---------------------------------------------------
        logger.info("Step 6: Fetching dropdown lists")

        async def get_all(model):
            result = await db.execute(select(model))
            rows = result.scalars().all()
            final = []
            for r in rows:
                d = r.__dict__.copy()
                d.pop("_sa_instance_state", None)
                if "id" in d and not isinstance(d["id"], str):
                    d["id"] = str(d["id"])
                final.append(d)
            return final

        therapeutic_area = await get_all(TherapeuticArea)
        topics = await get_all(VeevaTopics)
        audience = await get_all(VeevaAudience)
        segment = await get_all(VeevaSegment)
        uploader = await get_all(VeevaUploader)
        creative_agency = await get_all(VeevaCreativeAgency)
        permitted_channels = await get_all(PermittedChannels)
        permitted_countries = await get_all(PermittedCountries)

        # ---------------------------------------------------
        # STEP 7 — CHECK EXISTING LICENSEFILE
        # ---------------------------------------------------
        logger.info("Step 7: Checking existing LicenseFile")

        existing_q = await db.execute(
            select(LicenseFile).where(LicenseFile.image_id == str(image.id))
        )
        existing = existing_q.scalar_one_or_none()

        if existing:
            logger.info("Existing LicenseFile found → returning existing metadata")

            data = {
                "image_id": existing.image_id,
                "image_number":existing.image_number,
                "image_name": existing.name,
                "project_id": image.project_id,
                "product": existing.product,
                "country": existing.country,
                "classification": classification,
                "therapeutic_area": existing.therapeutic_area,
                "language": existing.language,
                "resolution": f"{existing.width} x {existing.height}",
                "format": existing.format,
                "name": existing.name,
                "subtitle": existing.subtitle,
                "s3_url": generate_presigned_url(existing.s3_url),
                "title": existing.title,
                "tags": existing.tags,
                "topic_options": existing.topic,
                "audience_options": existing.audience,
                "segment_options": existing.segment,
                "uploader_role_options": existing.uploader_role,
                "creative_agency_options": existing.creative_agency,
                "permitted_channel_options": existing.permitted_channels,
                "permitted_country_options": existing.permitted_countries,
                "is_license_generated": existing.is_license_generated,
                "license_doc": existing.license_doc,
                "descriptor":existing.descriptor,
                "default_fields": {
                    "license_start_date": existing.license_start_date,
                    "license_end_date": existing.license_end_date,
                    "rights": "All Generated Image",
                    "publish": True,
                    "rights_other__v": existing.rights_other__v
                }
            }

            return {"status": "success", "metadata_form": data}

        # ---------------------------------------------------
        # STEP 8 — CREATE NEW LICENSEFILE
        # ---------------------------------------------------
        logger.info("No LicenseFile exists → creating a new record")

        new_record = LicenseFile(
            image_id=str(image.id),
            veeva_id="",
            product=brand,
            image_number=str(image.image_name),
            country=country,
            tags=tags,
            therapeutic_area=therapeutic_area,
            language="Spanish",
            height=str(image.height),
            width=str(image.width),
            format="PNG",
            name=name,
            subtitle=subtitle,
            s3_url=image.s3_object_key,
            title=title,
            topic=topics,
            audience=audience,
            segment=segment,
            uploader_role=uploader,
            creative_agency=creative_agency,
            permitted_countries=permitted_countries,
            permitted_channels=permitted_channels,
            license_start_date=date.today().strftime("%Y-%m-%d"),
            license_end_date = date(2026, 11, 30).strftime("%Y-%m-%d"),
            rights_other__v="AI Generated Image",
            is_license_generated=False,
            license_doc="",
            descriptor=descriptor_string
        )

        db.add(new_record)
        await db.commit()

        # ---------------------------------------------------
        # STEP 9 — BUILD RESPONSE FOR NEW RECORD
        # ---------------------------------------------------
        logger.info("Building response for new LicenseFile")

        data = {
            "image_id": image.id,
            "image_name": name,
            "image_number":image.image_name,
            "project_id": image.project_id,
            "product": brand,
            "country": country,
            "classification": classification,
            "therapeutic_area": therapeutic_area,
            "language": "Spanish",
            "resolution": f"{image.width} x {image.height}",
            "format": "PNG",
            "name": name,
            "subtitle": subtitle,
            "s3_url": generate_presigned_url(image.s3_object_key),
            "title": title,
            "tags": tags,

            "topic_options": topics,
            "audience_options": audience,
            "segment_options": segment,
            "uploader_role_options": uploader,
            "creative_agency_options": creative_agency,
            "permitted_channel_options": permitted_channels,
            "permitted_country_options": permitted_countries,

            "is_license_generated": False,
            "license_doc": "",
            "descriptor":descriptor_string,
            "default_fields": {
                "license_start_date": date.today().strftime("%Y-%m-%d"),
                "license_end_date": date(2026, 11, 30).strftime("%Y-%m-%d"),
                "rights": "All Generated Image",
                "publish": True,
                "rights_other__v": "AI Generated Image"
            }
        }

        return {"status": "success", "metadata_form": data}
    


    async def veeva_plus_search(
        body: Dict[str, Any],
        vectorstore,
        async_client,
        db: AsyncSession,
        limit: int = 20,
        offset: int = 0,
    ):
        logger.info("\n===========================")
        logger.info("🚀 Starting /search endpoint")
        logger.info("===========================\n")

        # ----------------------------------------------------------
        # Filters
        # ----------------------------------------------------------
        filter_conditions = {}
        if body.get("country"):
            filter_conditions["geography"] = body["country"]
        if body.get("type"):
            filter_conditions["classification"] = body["type"]
        if body.get("product"):
            filter_conditions["brand"] = body["product"]

        final_filter = filter_conditions or None
        has_query = bool(body.get("query") and body["query"].strip())

        combined_results = []

        # ==========================================================
        # CASE 1 — SEMANTIC SEARCH
        # ==========================================================
        if has_query:
            logger.info("🧠 Mode: Semantic Search")

            veeva_results = []
            local_results = []

            # --- VEEVA VECTOR SEARCH ---
            try:
                veeva_results = vectorstore.similarity_search_with_score(
                    query=body["query"], filter=final_filter
                )
                logger.info(veeva_results)
            except Exception as e:
                logger.warning(f"⚠️ Veeva search failed: {e}")
            


            # --- LOCAL VECTOR SEARCH ---
            try:
                local_vs = PGVector(
                    connection_string=settings.SQLALCHEMY_DATABASE_SSL_URI,
                    collection_name="local_prompts",
                    embedding_function=embeddings
                )
                local_results = local_vs.similarity_search_with_score(
                    query=body["query"], filter=final_filter
                )
                logger.info(local_results)
            except Exception as e:
                logger.warning(f"⚠️ Local search failed: {e}")

            # ----------------------------------------------------------
            # Parse Veeva Results
            # ----------------------------------------------------------
            for result in veeva_results:
                
                try:
                    logger.info("This is item of veeva_result")
                    doc, dist = result
                    meta = getattr(doc, "metadata", {}) or {}
                    score = round(1 - float(dist), 3)
                    logger.info(score)
                    if score < 0.5:
                        continue
                    if not meta.get("s3_url"):
                        continue

                    date_val = (
                        _parse_date(meta.get("version_date"))
                        or _parse_date(meta.get("updated_at"))
                    )

                    combined_results.append({
                        "source": "veeva",
                        "id": meta.get("id"),
                        "veeva_id": meta.get("veeva_id"),
                        "brand": meta.get("brand"),
                        "country": meta.get("geography"),
                        "classification": meta.get("classification"),
                        "title": meta.get("title"),
                        "snippet": meta.get("snippet"),
                        "name":meta.get("name"),
                        "date": date_val,
                        "score": score,
                        "s3_url": generate_presigned_url(meta.get("s3_url")),
                        "published_document_number": meta.get("document_number"),
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Error parsing Veeva result: {e}")

            # ----------------------------------------------------------
            # Parse Local Results
            # ----------------------------------------------------------
            for result in local_results:
                try:
                    logger.info("This is a local result")
                    doc, dist = result
                    meta = getattr(doc, "metadata", {}) or {}
                    score = round(1 - float(dist), 3)
                    logger.info(score)
                    if score < 0.5:
                        continue

                    prompt_id = meta.get("prompt_id") or meta.get("id")
                    if not prompt_id:
                        continue

                    img_stmt = select(Image).where(Image.prompt_id == prompt_id)
                    imgs = (await db.execute(img_stmt)).scalars().all()

                    for img in imgs:
                        if not img.s3_object_key:
                            continue

                        # Fetch project
                        project = None
                        if img.project_id:
                            proj_stmt = select(Project).where(Project.id == img.project_id)
                            project = (await db.execute(proj_stmt)).scalars().first()

                        # fetch veeva_id if published
                        veeva_id = ""
                        if img.is_published:
                            vid_stmt = select(LicenseFile.veeva_id).where(
                                LicenseFile.image_id == str(img.id)
                            )
                            veeva_id = (await db.execute(vid_stmt)).scalar_one_or_none() or ""

                        combined_results.append({
                            "source": "local",
                            "id": str(img.id),
                            "prompt_id": prompt_id,
                            "veeva_id": veeva_id,
                            "is_published": img.is_published,
                            "brand": getattr(project, "product", None),
                            "country": getattr(project, "country", None),
                            "classification": getattr(project, "type", None),
                            "title": meta.get("prompt") or doc.page_content,
                            "image_name": img.image_name,
                            "score": score,
                            "date": img.updated_at,
                            "s3_url": generate_presigned_url(img.s3_object_key),
                            "published_document_number": img.veeva_published_number,
                            "published_document_status":img.veeva_published_status,
                        })

                except Exception as e:
                    logger.warning(f"⚠️ Local parse error: {e}")

        # ==========================================================
        # CASE 2 — METADATA SEARCH
        # ==========================================================
        else:
            logger.info("🗂️ Mode: Metadata Search")

            veeva_filters = []
            project_filters = []

            if body.get("product"):
                veeva_filters.append(VeevaDoc.brand.ilike(f"%{body['product']}%"))
                project_filters.append(Project.product.ilike(f"%{body['product']}%"))

            if body.get("country"):
                veeva_filters.append(VeevaDoc.geography.ilike(f"%{body['country']}%"))
                project_filters.append(Project.country.ilike(f"%{body['country']}%"))

            if body.get("type"):
                veeva_filters.append(VeevaDoc.classification.ilike(f"%{body['type']}%"))
                project_filters.append(Project.type.ilike(f"%{body['type']}%"))

            # --- VEEVA DOCS ---
            v_stmt = select(VeevaDoc)
            if veeva_filters:
                v_stmt = v_stmt.where(and_(*veeva_filters))
            v_stmt = v_stmt.order_by(desc(VeevaDoc.version_date))

            veeva_docs = (await db.execute(v_stmt)).scalars().all()

            for doc in veeva_docs:
                if not doc.s3_url:
                    continue

                combined_results.append({
                    "source": "veeva",
                    "id": str(doc.id),
                    "veeva_id": str(doc.veeva_id),
                    "brand": doc.brand,
                    "country": doc.geography,
                    "classification": doc.classification,
                    "title": doc.title,
                    "date": doc.version_date,
                    "s3_url": generate_presigned_url(doc.s3_url),
                    "published_document_number": doc.document_number,
                })

            # --- LOCAL IMAGES ---
            p_stmt = select(Project)
            if project_filters:
                p_stmt = p_stmt.where(and_(*project_filters))

            projects = (await db.execute(p_stmt)).scalars().all()
            project_ids = [p.id for p in projects] or [None]

            img_stmt = (
                select(Image)
                .where(Image.project_id.in_(project_ids))
                .order_by(desc(Image.updated_at))
            )
            images = (await db.execute(img_stmt)).scalars().all()

            for img in images:
                if not img.s3_object_key:
                    continue

                veeva_id = ""
                if img.is_published:
                    vid_stmt = select(LicenseFile.veeva_id).where(
                        LicenseFile.image_id == str(img.id)
                    )
                    veeva_id = (await db.execute(vid_stmt)).scalar_one_or_none() or ""

                combined_results.append({
                    "source": "local",
                    "id": str(img.id),
                    "veeva_id": veeva_id,
                    "brand": getattr(img.project, "product", None),
                    "country": getattr(img.project, "country", None),
                    "classification": getattr(img.project, "type", None),
                    "image_name": img.image_name,
                    "date": img.updated_at,
                    "s3_url": generate_presigned_url(img.s3_object_key),
                    "is_published": img.is_published,
                    "published_document_number": img.veeva_published_number,
                    "published_document_status":img.veeva_published_status,

                })

        # ==========================================================
        # ⭐ CONTEXTUAL RERANK — EXACT SAME SYSTEM PROMPT
        # ==========================================================
        async def context_rerank(query: str, items: list[dict]):
            system_prompt = (
                "You are a retrieval relevance evaluator.\n\n"
                "Score how well the CANDIDATE matches the QUERY based ONLY on true core meaning.\n"
                "Do NOT hallucinate required details. Ignore all visual, stylistic, descriptive, or enhancement differences.\n\n"
                "CORE MEANING RULE:\n"
                "- Identify the essential meaning of the QUERY (main subject(s), main object(s), main action).\n"
                "- If the CANDIDATE preserves this same meaning, assign a high score.\n"
                "- Extra harmless details (background color, lighting, clothing, pose, environment style) must NOT reduce the score.\n"
                "- Reduce the score ONLY when the CANDIDATE changes, removes, or replaces a meaning-critical element.\n"
                "- Penalize meaning conflicts: apple (fruit) vs Apple (phone), tablet device vs tablet pill.\n\n"
                "FULL SCORING SCALE (0.0 to 1.0):\n"
                "1.0 — Perfect match. Core meaning identical. No conflicts.\n"
                "0.9 — Near-perfect match. Same meaning with tiny harmless variation.\n"
                "0.8 — Very strong match. Meaning preserved; one small nuance missing.\n"
                "0.7 — Strong match. Meaning clear but slightly diluted.\n"
                "0.6 — Somewhat similar. Subject/object matches; action loosely related.\n"
                "0.5 — Moderately related. Shares core subject or object but not full action.\n"
                "0.4 — Weak relation. Same broad domain but core object/action missing.\n"
                "0.3 — Very weak relation. Only superficial theme matches.\n"
                "0.2 — Barely related. Meaning mostly different.\n"
                "0.1 — Almost unrelated. Only tiny vague overlap.\n"
                "0.0 — Completely unrelated OR contradictory meaning.\n"
                "      Also use 0.0 when a key term changes meaning (e.g., apple fruit → Apple phone).\n\n"
                "Return only a numeric score between 0 and 1."
            )

            scored_items = []

            for item in items:
                title = item.get("title") or item.get("image_name") or ""
                name = item.get("name")
                snippet = item.get("snippet") or ""
                text = f"Title:{title} and Name:{name}"

                try:
                    resp = await async_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content":
                                f"Query: {query}\nCandidate: {text}"}
                        ],
                        temperature=0,
                        max_completion_tokens=2000
                    )
                    print(resp)
                    cs = float(resp.choices[0].message.content.strip())
                except:
                    cs = 0.0

                base = item.get("score") or 0.0
                item["context_score"] = cs
                item["final_score"] = round(base * cs, 3)
                scored_items.append(item)

                logger.info("These are scored items")
                logger.info(scored_items)

            filtered = [i for i in scored_items if i["final_score"] >= 0.35]

            filtered.sort(key=lambda x: x["final_score"], reverse=True)

            return filtered

        if has_query and combined_results:
            combined_results = await context_rerank(body["query"], combined_results)

        # ==========================================================
        # ❌ REMOVE VEEVA DOCS IF LOCAL IS PUBLISHED
        # ==========================================================
        local_published_numbers = {
            item.get("published_document_number")
            for item in combined_results
            if item["source"] == "local" and item.get("is_published")
        }

        combined_results = [
            item for item in combined_results
            if not (
                item["source"] == "veeva"
                and item.get("published_document_number") in local_published_numbers
            )
        ]

        # ==========================================================
        # 🛠️ NORMALIZE DATES
        # ==========================================================
        for item in combined_results:
            val = item.get("date")

            if not isinstance(val, datetime):
                try:
                    val = _parse_date(val)
                except:
                    val = None

            if isinstance(val, datetime) and val.tzinfo is None:
                val = val.replace(tzinfo=timezone.utc)

            item["date"] = val or datetime.min.replace(tzinfo=timezone.utc)

        # ==========================================================
        # SORT + PAGINATE
        # ==========================================================
        combined_results.sort(
            key=lambda x: (
                x.get("date"),
                x.get("final_score", x.get("score", 0)),
            ),
            reverse=True
        )

        paginated = combined_results[offset:offset + limit]

        return {
            "total": len(combined_results),
            "limit": limit,
            "offset": offset,
            "results": paginated,
        }
    

#     async def run_ingestion(body: IngestRequest,veeva,vectorstore,db: AsyncSession):
#         """
#         Full ingestion logic moved from router.
#         Router ONLY calls this function.
#         """

#         logger.info("🚀 Starting ingestion logic inside service...")
#         logger.info(f"🔹 brand_id={body.brand_id}, classification_id={body.classification_id}, country={body.country}")

#         # ----------------------------------------
#         # 1️⃣ Fetch Brand Lookup
#         # ----------------------------------------
#         brand_row = await db.execute(
#             select(VeevaBrand).where(VeevaBrand.veeva_id == str(body.brand_id))
#         )
#         brand_obj = brand_row.scalars().first()
#         if not brand_obj:
#             raise HTTPException(400, f"Brand Veeva ID {body.brand_id} not found")

#         brand_veeva_id = brand_obj.veeva_id
#         brand_value = brand_obj.value
#         logger.info(f"🏷️ Brand → veeva_id={brand_veeva_id}, value={brand_value}")

#         # ----------------------------------------
#         # 2️⃣ Fetch Optional Classification
#         # ----------------------------------------
#         class_veeva_id, class_value = None, None

#         if body.classification_id and str(body.classification_id).strip() not in ["", "0", "None"]:
#             class_row = await db.execute(
#                 select(VeevaClassifications).where(VeevaClassifications.veeva_id == str(body.classification_id))
#             )
#             class_obj = class_row.scalars().first()
#             if not class_obj:
#                 raise HTTPException(400, f"Classification Veeva ID {body.classification_id} not found")

#             class_veeva_id = class_obj.veeva_id
#             class_value = class_obj.value
#             logger.info(f"📂 Classification → {class_value}")
#         else:
#             logger.info("⚙️ No classification filter used")

#         # ----------------------------------------
#         # 3️⃣ Build VQL Query
#         # ----------------------------------------
#         country_val = body.country

#         conditions = [f"product__v CONTAINS ('{brand_veeva_id}')"]
#         if class_value:
#             conditions.append(f"classification__v = '{class_value}'")
#         if country_val:
#             conditions.append(f"country__v CONTAINS ('{country_val}')")

#         conditions.append(f"status__v = 'Available for Use'")
#         where_clause = " AND ".join(conditions)

#         vql = f"""
# SELECT id, name__v, document_number__v, title__v, description__v, classification__v,
#        status__v, license_start_date__c, license_expiration_date__c,
#        is_rights_managed__c, country__v, audience_code__c, segment__c,
#        uploader_role__c, creative_agency1__c, tags__v, version_id, 
#        version_modified_date__v, rights_other__v
# FROM documents
# WHERE {where_clause}
# """.strip()

#         logger.info("📜 VQL Query Prepared")
#         logger.info(vql)

#         # ----------------------------------------
#         # 4️⃣ Fetch Records from Veeva
#         # ----------------------------------------
#         session = body.session_id
#         rows = veeva.query_vql(session, vql)
#         logger.info(f"📦 Rows fetched: {len(rows)}")

#         # Prepare embedding lists
#         docs, metadatas, vec_ids, delete_ids = [], [], [], []
#         count = 0

#         # A separate fresh session for renditions
#         rendition_session = veeva.veeva_auth()

#         # ----------------------------------------
#         # 5️⃣ Process Each Document
#         # ----------------------------------------
#         for r in rows:
#             veeva_id = str(r.get("id") or "").strip()
#             id = uuid.uuid4()            # UUID object

#             if not veeva_id:
#                 logger.warning("⚠️ Missing ID — Skipping doc")
#                 continue

#             logger.info(f"📝 Processing document {veeva_id}")

#             # ----------------------------------------
#             # Renditions → S3 Upload
#             # ----------------------------------------
#             try:
#                 s3_url = veeva.get_doc_rendition(rendition_session, doc_id=veeva_id)
#                 logger.info(f"🖼️ Rendition saved → {s3_url}")
#             except Exception as e:
#                 logger.warning(f"⚠️ Rendition fetch failed for {veeva_id}: {str(e)}")
#                 s3_url = None

#             # ----------------------------------------
#             # Extract Metadata
#             # ----------------------------------------
#             title_val = (r.get("title__v") or "").strip()
#             name_val = (r.get("name__v") or "").strip()
#             document_number_val = r.get("document_number__v")
#             snippet_val = (r.get("description__v") or "").strip()

#             status_val = r.get("status__v") or r.get("status__c")
#             license_start = r.get("license_start_date__c")
#             license_end = r.get("license_expiration_date__c")
#             rights_managed = r.get("is_rights_managed__c")

#             # List-normalization
#             ensure_list = lambda v: [v] if isinstance(v, str) else v
#             country_list = ensure_list(r.get("country__v"))
#             audience_code = ensure_list(r.get("audience_code__c"))
#             uploader_role = ensure_list(r.get("uploader_role__c"))
#             creative_agency = ensure_list(r.get("creative_agency1__c"))
#             tags = ensure_list(r.get("tags__v"))
#             segment_val = r.get("segment__c")
#             version_id = r.get("version_id")
#             version_date = r.get("version_modified_date__v")
#             rights_other = r.get("rights_other__v") or "Veeva Publish"

#             # ----------------------------------------
#             # Check DB for Existing Document
#             # ----------------------------------------
#             db_row = await db.execute(select(VeevaDoc).where(VeevaDoc.veeva_id == veeva_id))
#             db_doc = db_row.scalars().first()

#             reembed = False

#             if not db_doc:
#                 logger.info("➕ Inserting new VeevaDoc row")
#                 reembed = True

#                 db_doc = VeevaDoc(
#                     id=id,
#                     veeva_id=veeva_id,
#                     brand=brand_value,
#                     geography=country_val,
#                     classification=class_value,
#                     document_number=document_number_val,
#                     name=name_val,
#                     title=title_val,
#                     snippet=snippet_val,
#                     status=status_val,
#                     license_start_date=license_start,
#                     license_expiration_date=license_end,
#                     is_rights_managed=rights_managed,
#                     country=country_list,
#                     audience_code=audience_code,
#                     segment=segment_val,
#                     uploader_role=uploader_role,
#                     creative_agency=creative_agency,
#                     tags=tags,
#                     version_id=version_id,
#                     s3_url=s3_url,
#                     version_date=version_date,
#                     rights_other__v=rights_other
#                 )
#                 db.add(db_doc)

#             else:
#                 logger.info("🔄 Updating existing document")

#                 # Detect text change
#                 text_changed = (
#                     (db_doc.title or "").strip() != title_val or
#                     (db_doc.name or "").strip() != name_val or
#                     (db_doc.snippet or "").strip() != snippet_val
#                 )

#                 if text_changed:
#                     logger.info("🟠 Re-embedding required")
#                     reembed = True
#                 else:
#                     logger.info("🟢 No text change → Skipping re-embed")

#                 # Always update metadata
#                 db_doc.title = title_val
#                 db_doc.name = name_val
#                 db_doc.snippet = snippet_val
#                 db_doc.brand = brand_value
#                 db_doc.geography = country_val
#                 db_doc.classification = class_value
#                 db_doc.status = status_val
#                 db_doc.license_start_date = license_start
#                 db_doc.license_expiration_date = license_end
#                 db_doc.is_rights_managed = rights_managed
#                 db_doc.country = country_list
#                 db_doc.audience_code = audience_code
#                 db_doc.segment = segment_val
#                 db_doc.uploader_role = uploader_role
#                 db_doc.creative_agency = creative_agency
#                 db_doc.tags = tags
#                 db_doc.version_id = version_id
#                 db_doc.s3_url = s3_url
#                 db_doc.version_date = version_date
#                 db_doc.document_number = document_number_val
#                 db_doc.rights_other__v = rights_other

#             # ----------------------------------------
#             # EMBEDDING HANDLING
#             # ----------------------------------------
#             if reembed and s3_url:
#                 doc_uid = f"veeva::{veeva_id}"
#                 delete_ids.append(doc_uid)

#                 docs.append(
#                     f"Title: {title_val} | Name: {name_val}\n\n{snippet_val}"
#                 )

#                 metadatas.append({
#                     "id":str(id),
#                     "veeva_id": veeva_id,
#                     "title": title_val,
#                     "name": name_val,
#                     "snippet": snippet_val,
#                     "brand": brand_value.lower(),
#                     "geography": country_val.lower(),
#                     "classification": class_value.lower() if class_value else None,
#                     "status": status_val,
#                     "license_start_date": license_start,
#                     "license_expiration_date": license_end,
#                     "is_rights_managed": rights_managed,
#                     "country": country_list,
#                     "audience_code": audience_code,
#                     "segment": segment_val,
#                     "uploader_role": uploader_role,
#                     "creative_agency": creative_agency,
#                     "tags": tags,
#                     "version_id": version_id,
#                     "s3_url": s3_url,
#                     "version_date": version_date,
#                     "document_number": document_number_val,
#                     "rights_other__v": rights_other,
#                 })

#                 vec_ids.append(doc_uid)

#             count += 1
#             await asyncio.sleep(0.05)

#         # ----------------------------------------
#         # 6️⃣ Sync Vectorstore
#         # ----------------------------------------
#         if delete_ids:
#             logger.info(f"🗑️ Deleting {len(delete_ids)} old embeddings...")
#             try:
#                 vectorstore.delete(ids=delete_ids)
#             except Exception:
#                 logger.warning("Delete failed, retrying one-by-one…")
#                 for did in delete_ids:
#                     vectorstore.delete(filter={"id": did.split("::")[1]})

#         if docs:
#             logger.info(f"🧠 Adding {len(docs)} embeddings...")
#             vectorstore.add_texts(docs, metadatas=metadatas, ids=vec_ids)

#         # ----------------------------------------
#         # 7️⃣ Commit to DB
#         # ----------------------------------------
#         await db.commit()
#         logger.info(f"✅ Ingestion finished — {count} documents processed")

#         return count
    

    async def run_ingest_local(body: IngestRequest, db: AsyncSession):
        """
        Full logic for ingesting local prompts into PGVector.
        Router remains clean.
        """

        logger.info("🚀 Starting run_ingest_local service method")

        # ----------------------------------------
        # Initialize vectorstore for local_prompts
        # ----------------------------------------
        local_vectorstore = PGVector(
            connection_string=settings.SQLALCHEMY_DATABASE_SSL_URI,
            collection_name="local_prompts",
            embedding_function=embeddings,
            use_jsonb=True,
        )

        # ----------------------------------------
        # Build query for prompts
        # ----------------------------------------
        stmt = select(Prompt)

        if body.created_date:
            try:
                created_date = datetime.strptime(body.created_date, "%Y-%m-%d").date()
                stmt = stmt.where(func.date(Prompt.created_at) >= created_date)
                logger.info(f"📅 Filtering prompts created after: {created_date}")
            except ValueError:
                logger.error("❌ Invalid created_date format. Expected YYYY-MM-DD.")
                raise HTTPException(400, "Invalid created_date format")

        result = await db.execute(stmt)
        prompts = result.scalars().all()

        logger.info(f"📦 Found {len(prompts)} prompts for ingestion")

        # ----------------------------------------
        # Prepare tracking containers
        # ----------------------------------------
        docs, metadatas, ids, delete_ids = [], [], [], []
        count = 0

        # ----------------------------------------
        # Loop through prompts
        # ----------------------------------------
        for p in prompts:
            if not (p.prompt or p.enhanced_prompt):
                continue

            # Combine prompt + enhanced_prompt
            text_for_embedding = ""
            if p.prompt:
                text_for_embedding += p.prompt.strip()
            

            doc_uid = f"prompt::{p.id}"
            ids.append(doc_uid)
            delete_ids.append(doc_uid)

            # ----------------------------------------
            # Fetch brand / geography / type
            # via Image → Project linkage
            # ----------------------------------------
            brand = geography = classification = None

            img_stmt = select(Image).where(Image.prompt_id == p.id).limit(1)
            img_res = await db.execute(img_stmt)
            image = img_res.scalar_one_or_none()

            if image and image.project_id:
                proj_stmt = select(Project).where(Project.id == image.project_id)
                proj_res = await db.execute(proj_stmt)
                project = proj_res.scalar_one_or_none()

                if project:
                    brand = project.product
                    geography = project.country
                    classification = project.type

            # ----------------------------------------
            # Build metadata JSON
            # ----------------------------------------
            metadatas.append({
                "id": str(p.id),
                "prompt": p.prompt,
                "enhanced_prompt": p.enhanced_prompt,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "user_id": p.user_id,
                "brand": brand,
                "geography": geography,
                "classification": classification
            })

            docs.append(text_for_embedding)
            count += 1

            await asyncio.sleep(0.03)

        # ----------------------------------------
        # Delete old embeddings
        # ----------------------------------------
        if delete_ids:
            logger.info(f"🧹 Deleting {len(delete_ids)} old embeddings from local_prompts")
            try:
                local_vectorstore.delete(ids=delete_ids)
            except Exception:
                logger.warning("⚠️ Bulk delete failed — retrying individually")
                for did in delete_ids:
                    local_vectorstore.delete(filter={"id": did.split('::')[1]})

        # ----------------------------------------
        # Insert fresh embeddings
        # ----------------------------------------
        if docs:
            logger.info(f"🧠 Adding {len(docs)} new embeddings into local_prompts")
            local_vectorstore.add_texts(docs, metadatas=metadatas, ids=ids)

        # Commit DB session
        await db.commit()

        logger.info(f"✅ Local ingestion completed: {count} prompts processed")
        return count
    


    async def run_ingestion(body: IngestRequest, veeva, vectorstore, db: AsyncSession):
        """
        Clean rewritten ingestion flow inside a class method.
        Embeddings now happen per-document (real-time), not batched.
        """
        logger.info("🚀 Starting ingestion logic inside service...")
        logger.info(f"🔹 brand_id={body.brand_id}, classification_id={body.classification_id}, country={body.country}")

        # ----------------------------------------
        # 1️⃣ Fetch Brand Object
        # ----------------------------------------
        brand_row = await db.execute(
            select(VeevaBrand).where(VeevaBrand.veeva_id == str(body.brand_id))
        )
        brand_obj = brand_row.scalars().first()

        if not brand_obj:
            raise HTTPException(400, f"Brand Veeva ID {body.brand_id} not found")

        brand_veeva_id = brand_obj.veeva_id
        brand_value = brand_obj.value
        logger.info(f"🏷️ Brand → {brand_value}")

        # ----------------------------------------
        # 2️⃣ Fetch Classification (optional)
        # ----------------------------------------
        class_veeva_id, class_value = None, None

        if body.classification_id and str(body.classification_id).strip() not in ["", "0", "None"]:
            class_row = await db.execute(
                select(VeevaClassifications).where(VeevaClassifications.veeva_id == str(body.classification_id))
            )
            class_obj = class_row.scalars().first()

            if not class_obj:
                raise HTTPException(400, f"Classification Veeva ID {body.classification_id} not found")

            class_veeva_id = class_obj.veeva_id
            class_value = class_obj.value
            logger.info(f"📂 Classification → {class_value}")
        else:
            logger.info("⚙️ No classification filter used")

        # ----------------------------------------
        # 3️⃣ VQL Query
        # ----------------------------------------
        country_val = body.country

        conditions = [f"product__v CONTAINS ('{brand_veeva_id}')"]

        if class_value:
            conditions.append(f"classification__v = '{class_value}'")
        if country_val:
            conditions.append(f"country__v CONTAINS ('{country_val}')")

        conditions.append("status__v = 'Available for Use'")
        where_clause = " AND ".join(conditions)

        vql = f"""
SELECT id, name__v, document_number__v, title__v, description__v,
       classification__v, status__v, license_start_date__c, license_expiration_date__c,
       is_rights_managed__c, country__v, audience_code__c, segment__c,
       uploader_role__c, creative_agency1__c, tags__v,
       version_id, version_modified_date__v, rights_other__v
FROM documents
WHERE {where_clause}
""".strip()

        logger.info("📜 VQL Query:")
        logger.info(vql)

        # ----------------------------------------
        # 4️⃣ Fetch Records
        # ----------------------------------------
        session = body.session_id
        rows = veeva.query_vql(session, vql)
        logger.info(f"📦 Rows fetched: {len(rows)}")

        rendition_session = veeva.veeva_auth()
        count = 0

        # ----------------------------------------
        # Helper to force lists
        # ----------------------------------------
        def ensure_list(v):
            if v is None:
                return []
            if isinstance(v, list):
                return v
            return [v]

        # ----------------------------------------
        # 5️⃣ PROCESS EACH DOCUMENT
        # ----------------------------------------
        for r in rows:

            veeva_id = str(r.get("id") or "").strip()
            if not veeva_id:
                logger.warning("⚠️ Missing Veeva ID — skipping")
                continue

            logger.info(f"📝 Processing VeevaDoc {veeva_id}")

            # Fetch rendition → S3
            try:
                s3_url = veeva.get_doc_rendition(rendition_session, doc_id=veeva_id)
                logger.info(f"🖼️ Rendition saved → {s3_url}")
            except Exception as e:
                logger.warning(f"⚠️ Rendition failed for {veeva_id}: {e}")
                s3_url = None

            # Metadata
            title_val = (r.get("title__v") or "").strip()
            name_val = (r.get("name__v") or "").strip()
            snippet_val = (r.get("description__v") or "").strip()
            document_number_val = r.get("document_number__v")

            status_val = r.get("status__v") or r.get("status__c")
            license_start = r.get("license_start_date__c")
            license_end = r.get("license_expiration_date__c")
            rights_other = r.get("rights_other__v") or "Veeva Publish"

            country_list = ensure_list(r.get("country__v"))
            audience_code = ensure_list(r.get("audience_code__c"))
            uploader_role = ensure_list(r.get("uploader_role__c"))
            creative_agency = ensure_list(r.get("creative_agency1__c"))
            tags_list = ensure_list(r.get("tags__v"))

            version_id = r.get("version_id")
            version_date = r.get("version_modified_date__v")
            segment_val = r.get("segment__c")
            rights_managed = r.get("is_rights_managed__c")

            # ----------------------------------------
            # Check DB existence
            # ----------------------------------------
            db_row = await db.execute(select(VeevaDoc).where(VeevaDoc.veeva_id == veeva_id))
            db_doc = db_row.scalars().first()

            reembed = False

            # ----------------------------------------
            # NEW DOCUMENT
            # ----------------------------------------
            if not db_doc:
                logger.info("➕ New document → inserting")
                reembed = True

                new_id = uuid.uuid4()

                db_doc = VeevaDoc(
                    id=new_id,
                    veeva_id=veeva_id,
                    brand=brand_value,
                    geography=country_val,
                    classification=class_value,
                    document_number=document_number_val,
                    name=name_val,
                    title=title_val,
                    snippet=snippet_val,
                    status=status_val,
                    license_start_date=license_start,
                    license_expiration_date=license_end,
                    is_rights_managed=rights_managed,
                    country=country_list,
                    audience_code=audience_code,
                    segment=segment_val,
                    uploader_role=uploader_role,
                    creative_agency=creative_agency,
                    tags=tags_list,
                    version_id=version_id,
                    s3_url=s3_url,
                    version_date=version_date,
                    rights_other__v=rights_other
                )

                db.add(db_doc)

            # ----------------------------------------
            # EXISTING DOCUMENT
            # ----------------------------------------
            else:
                logger.info("🔄 Document exists → updating")

                text_changed = (
                    (db_doc.title or "").strip() != title_val or
                    (db_doc.name or "").strip() != name_val or
                    (db_doc.snippet or "").strip() != snippet_val
                )

                if text_changed:
                    logger.info("🟠 Text changed → re-embedding required")
                    reembed = True

                # Always update metadata
                db_doc.title = title_val
                db_doc.name = name_val
                db_doc.snippet = snippet_val
                db_doc.brand = brand_value
                db_doc.geography = country_val
                db_doc.classification = class_value
                db_doc.status = status_val
                db_doc.license_start_date = license_start
                db_doc.license_expiration_date = license_end
                db_doc.is_rights_managed = rights_managed
                db_doc.country = country_list
                db_doc.audience_code = audience_code
                db_doc.segment = segment_val
                db_doc.uploader_role = uploader_role
                db_doc.creative_agency = creative_agency
                db_doc.tags = tags_list
                db_doc.version_id = version_id
                db_doc.s3_url = s3_url
                db_doc.version_date = version_date
                db_doc.document_number = document_number_val
                db_doc.rights_other__v = rights_other

            # ----------------------------------------
            # 6️⃣ REAL-TIME EMBEDDING (PER DOCUMENT)
            # ----------------------------------------
            if reembed and s3_url:

                doc_uid = f"veeva::{veeva_id}"

                # Delete existing embedding (safe)
                try:
                    vectorstore.delete(ids=[doc_uid])
                except Exception:
                    try:
                        vectorstore.delete(filter={"veeva_id": veeva_id})
                    except:
                        logger.warning(f"⚠️ Could not delete embedding for {veeva_id}")

                text_to_embed = f"Title: {title_val} | Name: {name_val}\n\n{snippet_val}"

                metadata = {
                    "id": str(db_doc.id),
                    "veeva_id": veeva_id,
                    "title": title_val,
                    "name": name_val,
                    "snippet": snippet_val,
                    "brand": brand_value.lower(),
                    "geography": country_val.lower(),
                    "classification": class_value.lower() if class_value else None,
                    "status": status_val,
                    "license_start_date": license_start,
                    "license_expiration_date": license_end,
                    "is_rights_managed": rights_managed,
                    "country": country_list,
                    "audience_code": audience_code,
                    "segment": segment_val,
                    "uploader_role": uploader_role,
                    "creative_agency": creative_agency,
                    "tags": tags_list,
                    "version_id": version_id,
                    "s3_url": s3_url,
                    "version_date": version_date,
                    "document_number": document_number_val,
                    "rights_other__v": rights_other
                }

                # Insert embedding instantly
                try:
                    vectorstore.add_texts(
                        texts=[text_to_embed],
                        metadatas=[metadata],
                        ids=[doc_uid]
                    )
                    logger.info(f"🧠 Embedded → {veeva_id}")
                except Exception as e:
                    logger.error(f"❌ Embedding failed for {veeva_id}: {e}")

            count += 1
            await asyncio.sleep(0.05)

        # ----------------------------------------
        # 7️⃣ FINAL COMMIT
        # ----------------------------------------
        await db.commit()
        logger.info(f"✅ Ingestion finished — {count} documents processed")

        return count
  
    async def run_ingestion_specific(body: IngestRequest, veeva, vectorstore, db: AsyncSession):
        """
        Ingest ONLY specific document IDs from body.list.
        No brand / classification / country filters.
        Only embed documents with status = 'Available for Use'.
        """

        logger.info("🚀 Starting ID-specific ingestion...")
        logger.info(f"🔹 Document list = {body.list}")

        brand_row = await db.execute(
            select(VeevaBrand).where(VeevaBrand.veeva_id == str(body.brand_id))
        )
        brand_obj = brand_row.scalars().first()

        if not brand_obj:
            raise HTTPException(400, f"Brand Veeva ID {body.brand_id} not found")

        brand_veeva_id = brand_obj.veeva_id
        brand_value = brand_obj.value

        if not body.list or len(body.list) == 0:
            raise HTTPException(400, "list cannot be empty")

        rendition_session = veeva.veeva_auth()
        total_processed = 0

        # Helper for list normalization
        def ensure_list(value):
            if value is None:
                return []
            if isinstance(value, list):
                return value
            return [value]

        # ----------------------------------------------------
        # LOOP THROUGH EACH DOCUMENT ID ONE BY ONE
        # ----------------------------------------------------
        for doc_id in body.list:

            logger.info(f"\n=============================")
            logger.info(f"🔍 Processing ID: {doc_id}")
            logger.info(f"=============================\n")

            # Build simple VQL
            vql = f"""
SELECT id, name__v, document_number__v, title__v, description__v,
       classification__v, status__v, license_start_date__c, license_expiration_date__c,
       is_rights_managed__c, country__v, audience_code__c, segment__c,
       uploader_role__c, creative_agency1__c, tags__v,
       version_id, version_modified_date__v, rights_other__v, product__v
FROM documents
WHERE id = '{doc_id}'
""".strip()

            logger.info("📜 VQL Query:")
            logger.info(vql)

            # Fetch record
            session = body.session_id
            rows = veeva.query_vql(session, vql)

            if not rows or len(rows) == 0:
                logger.warning(f"⚠️ No record found for {doc_id}, skipping")
                continue

            r = rows[0]

            # Read status
            status_val = r.get("status__v") or r.get("status__c")

            if status_val != "Available for Use":
                logger.warning(f"⛔ Skipping {doc_id} because status is NOT Available for Use → {status_val}")
                continue

            logger.info(f"✅ Status OK for {doc_id}, proceeding with ingestion")

            # ---------------------------------------------
            # FETCH METADATA
            # ---------------------------------------------
            title_val = (r.get("title__v") or "").strip()
            name_val = (r.get("name__v") or "").strip()
            snippet_val = (r.get("description__v") or "").strip()
            document_number_val = r.get("document_number__v")

            license_start = r.get("license_start_date__c")
            license_end = r.get("license_expiration_date__c")
            rights_other = r.get("rights_other__v") or "Veeva Publish"

            country_list = ensure_list(r.get("country__v"))
            audience_code = ensure_list(r.get("audience_code__c"))
            uploader_role = ensure_list(r.get("uploader_role__c"))
            creative_agency = ensure_list(r.get("creative_agency1__c"))
            tags_list = ensure_list(r.get("tags__v"))

            version_id = r.get("version_id")
            version_date = r.get("version_modified_date__v")
            segment_val = r.get("segment__c")
            rights_managed = r.get("is_rights_managed__c")


            # ---------------------------------------------
            # FETCH RENDITION → S3
            # ---------------------------------------------
            try:
                s3_url = veeva.get_doc_rendition(rendition_session, doc_id=doc_id)
                logger.info(f"🖼️ Rendition uploaded → {s3_url}")
            except Exception as e:
                logger.warning(f"⚠️ Rendition failed for {doc_id}: {e}")
                s3_url = None

            # ---------------------------------------------
            # CHECK IF EXISTS IN DB
            # ---------------------------------------------
            db_row = await db.execute(
                select(VeevaDoc).where(VeevaDoc.veeva_id == str(doc_id)
            ))
            db_doc = db_row.scalars().first()

            reembed = False

            # NEW DOCUMENT
            if not db_doc:

                logger.info(f"➕ Creating new DB entry for {doc_id}")
                reembed = True

                new_uid = uuid.uuid4()

                db_doc = VeevaDoc(
                    id=new_uid,
                    veeva_id=str(doc_id),
                    brand=brand_value,
                    geography=None,
                    classification=None,
                    document_number=document_number_val,
                    name=name_val,
                    title=title_val,
                    snippet=snippet_val,
                    status=status_val,
                    license_start_date=license_start,
                    license_expiration_date=license_end,
                    is_rights_managed=rights_managed,
                    country=country_list,
                    audience_code=audience_code,
                    segment=segment_val,
                    uploader_role=uploader_role,
                    creative_agency=creative_agency,
                    tags=tags_list,
                    version_id=version_id,
                    s3_url=s3_url,
                    version_date=version_date,
                    rights_other__v=rights_other,
                )

                db.add(db_doc)

            else:

                logger.info(f"🔄 Updating existing DB entry for {doc_id}")

                text_changed = (
                    (db_doc.title or "").strip() != title_val
                    or (db_doc.name or "").strip() != name_val
                    or (db_doc.snippet or "").strip() != snippet_val
                )

                if text_changed:
                    logger.info("🟠 Text changed → re-embedding required")
                    reembed = True

                # Update metadata
                db_doc.title = title_val
                db_doc.name = name_val
                db_doc.snippet = snippet_val
                db_doc.status = status_val
                db_doc.license_start_date = license_start
                db_doc.license_expiration_date = license_end
                db_doc.is_rights_managed = rights_managed
                db_doc.country = country_list
                db_doc.audience_code = audience_code
                db_doc.segment = segment_val
                db_doc.uploader_role = uploader_role
                db_doc.creative_agency = creative_agency
                db_doc.tags = tags_list
                db_doc.version_id = version_id
                db_doc.s3_url = s3_url
                db_doc.version_date = version_date
                db_doc.document_number = document_number_val
                db_doc.rights_other__v = rights_other

            # ---------------------------------------------
            # EMBEDDING
            # ---------------------------------------------
            if reembed and s3_url:

                doc_uid = f"veeva::{doc_id}"

                try:
                    vectorstore.delete(ids=[doc_uid])
                except Exception:
                    logger.warning(f"⚠️ Could not delete old embedding for {doc_id}")

                text_to_embed = f"Title: {title_val} | Name: {name_val}\n\n{snippet_val}"

                metadata = {
                    "id": str(db_doc.id),
                    "veeva_id": str(doc_id),
                    "title": title_val,
                    "name": name_val,
                    "snippet": snippet_val,
                    "brand": (brand_value or "").lower(),
                    "status": status_val,
                    "license_start_date": license_start,
                    "license_expiration_date": license_end,
                    "is_rights_managed": rights_managed,
                    "country": country_list,
                    "audience_code": audience_code,
                    "segment": segment_val,
                    "uploader_role": uploader_role,
                    "creative_agency": creative_agency,
                    "tags": tags_list,
                    "version_id": version_id,
                    "s3_url": s3_url,
                    "version_date": version_date,
                    "document_number": document_number_val,
                    "rights_other__v": rights_other
                }

                try:
                    vectorstore.add_texts(
                        texts=[text_to_embed],
                        metadatas=[metadata],
                        ids=[doc_uid]
                    )
                    logger.info(f"🧠 Embedded → {doc_id}")
                except Exception as e:
                    logger.error(f"❌ Embedding error for {doc_id}: {e}")

            total_processed += 1
            await asyncio.sleep(0.05)

        # COMMIT AT END
        await db.commit()

        logger.info(f"✅ Ingestion complete — {total_processed} documents processed")

        return total_processed