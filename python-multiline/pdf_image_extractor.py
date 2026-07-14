import fitz # PyMuPDF
import hashlib
import io
import base64

def extract_pdf_images(pdf_bytes: bytes, min_width=200, min_height=150, render_pages=False, render_dpi=120, max_images=400):
    """
    Extracts figure-like images from a PDF, dropping tiny icons/logos based on min dimensions.
    Deduplicates identical images across pages.
    Optionally falls back to rendering the whole page if no embedded raster images are found (useful for vector figures).
    """
    doc = fitz.open("pdf", pdf_bytes)
    images_metadata = []
    seen_hashes = set()

    for pno in range(len(doc)):
        if len(images_metadata) >= max_images:
            break
        page = doc.load_page(pno)
        seen_xrefs = set()

        # --- 1) embedded raster images (high quality) ---
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            
            base_image = doc.extract_image(xref)
            if not base_image: continue

            w, h = base_image["width"], base_image["height"]
            if w < min_width or h < min_height:
                continue # Skip small UI elements/logos

            img_bytes = base_image["image"]
            hsh = hashlib.md5(img_bytes).hexdigest()
            if hsh in seen_hashes:
                continue # Deduplicate
            seen_hashes.add(hsh)

            # Generate a tiny thumbnail to send immediately (saves UI memory)
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                img.thumbnail((100, 100))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                thumb_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
            except Exception as e:
                print(f"Thumb generation failed: {e}")
                thumb_b64 = ""

            images_metadata.append({
                "id": f"emb-{pno}-{xref}",
                "page": pno + 1,
                "width": w,
                "height": h,
                "source": "embedded",
                "thumb": f"data:image/png;base64,{thumb_b64}"
            })

        # --- 2) Optional fallback: render page if vector figures are suspected ---
        if render_pages and len(seen_xrefs) == 0:
            pix = page.get_pixmap(dpi=render_dpi) # Render vector graphics
            w, h = pix.w, pix.h
            if w < min_width or h < min_height:
                continue
                
            img_bytes = pix.tobytes("png")
            hsh = hashlib.md5(img_bytes).hexdigest()
            if hsh in seen_hashes: continue
            seen_hashes.add(hsh)
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                img.thumbnail((100, 100))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                thumb_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
            except Exception as e:
                print(f"Thumb generation failed: {e}")
                thumb_b64 = ""

            images_metadata.append({
                "id": f"ren-{pno}",
                "page": pno + 1,
                "width": w,
                "height": h,
                "source": "page",
                "thumb": f"data:image/png;base64,{thumb_b64}"
            })

    return {"count": len(images_metadata), "images": images_metadata}


def extract_single_image(pdf_bytes: bytes, image_id: str) -> bytes:
    """Fetch the full-res PNG bytes for a specific extracted figure."""
    doc = fitz.open("pdf", pdf_bytes)
    mode, pno_str, *rest = image_id.split("-")
    pno = int(pno_str)
    
    if mode == "emb":
        xref = int(rest[0])
        base_image = doc.extract_image(xref)
        # Ensure we return valid PNG bytes
        try:
            pix = fitz.Pixmap(base_image["colorspace"], base_image["samples"])
        except:
            pix = fitz.Pixmap(doc, xref)
        if pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        return pix.tobytes("png")
    
    elif mode == "ren":
        page = doc.load_page(pno)
        pix = page.get_pixmap(dpi=200)
        return pix.tobytes("png")
        
    return b""
