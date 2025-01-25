import os
import re

PAGE_MARKER_REGEX = re.compile(r"\[PAGE\s+(\d+)\]")


def load_pdf_pages_into_dict(text_file_path: str):
    """
    Returns a dict: { page_num_str: "the text for that page" }
    """
    with open(text_file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    parts = PAGE_MARKER_REGEX.split(text_data)

    pages_dict = {}
    for i in range(1, len(parts), 2):
        page_num = parts[i].strip()
        page_text = parts[i + 1].strip()
        pages_dict[page_num] = page_text

    return pages_dict


IMAGE_FILE_REGEX = re.compile(
    r"^(?P<base>.+)_page_(?P<page>\d+)_img_(?P<img_idx>\d+)\.txt$",
    re.IGNORECASE
)


def load_image_captions(captions_dir: str):
    """
    Returns a dict of dicts:
        { <basename>: { <page_str>: [list_of_captions_for_that_page] } }
    """
    data = {}

    for fname in os.listdir(captions_dir):
        if not fname.lower().endswith(".txt"):
            continue
        m = IMAGE_FILE_REGEX.match(fname)
        if not m:
            continue
        base = m.group("base")
        page = m.group("page")
        captions_path = os.path.join(captions_dir, fname)
        with open(captions_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        if base not in data:
            data[base] = {}
        if page not in data[base]:
            data[base][page] = []
        data[base][page].append(caption)

    return data


def merge_text_and_captions(page_texts: dict, base: str, all_captions: dict):
    """
    page_texts: {page_num_str: "page text"}
    base: The 'basename' for that PDF
    all_captions: { <base>: { <page_str>: [caption1, caption2,...] }, ... }

    Returns a dict { page_num_str: merged_text_with_captions } 
    """
    merged = {}
    for pnum, text_content in page_texts.items():
        combined = text_content
        if base in all_captions and pnum in all_captions[base]:
            for cap in all_captions[base][pnum]:
                combined += f"\n\n[IMAGE CAPTION] {cap}"
        merged[pnum] = combined
    return merged


def recombine_merged_pages(merged_dict):
    """
    Turn { '1': "...", '2': "...", ... } back into a single string with
    [PAGE 1] ... text ... [PAGE 2] ... text ...
    """
    out_str = []
    for page_str in sorted(merged_dict.keys(), key=lambda x: int(x)):
        page_content = merged_dict[page_str]
        out_str.append(f"[PAGE {page_str}]\n{page_content}\n")
    return "\n".join(out_str)


def main():
    texts_dir = "src/ingestion/data/processed/texts"
    captions_dir = "src/ingestion/data/processed/image_texts"
    output_dir = "src/ingestion/data/processed/merged"

    os.makedirs(output_dir, exist_ok=True)

    all_captions = load_image_captions(captions_dir)

    for txt_file in os.listdir(texts_dir):
        if not txt_file.lower().endswith(".txt"):
            continue

        base = os.path.splitext(txt_file)[0]

        text_path = os.path.join(texts_dir, txt_file)
        pages_dict = load_pdf_pages_into_dict(text_path)

        merged_pages_dict = merge_text_and_captions(pages_dict, base, all_captions)

        final_merged_text = recombine_merged_pages(merged_pages_dict)

        out_path = os.path.join(output_dir, f"{base}_merged.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_merged_text)

        print(f"Saved merged text+captions => {out_path}")


if __name__ == "__main__":
    main()
