#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import csv
import argparse
import sys
from pathlib import Path

def tmx_to_csv(tmx_file: str, csv_file: str, source_lang: str, target_lang: str):
    tree = ET.parse(tmx_file)
    root = tree.getroot()

    # TMX namespace handling
    namespace = {'tmx': 'http://www.lisa.org/tmx14'}
    if root.tag == "tmx":
        namespace = {}

    tu_elements = root.findall('.//tu', namespace)
    pairs = []

    for tu in tu_elements:
        source_text = target_text = None
        for tuv in tu.findall('tuv', namespace):
            lang = tuv.attrib.get('{http://www.w3.org/XML/1998/namespace}lang') or tuv.attrib.get('lang')
            seg = tuv.find('seg', namespace)
            if seg is None:
                continue
            text = seg.text.strip() if seg.text else ""
            if lang == source_lang:
                source_text = text
            elif lang == target_lang:
                target_text = text
        if source_text and target_text:
            pairs.append((source_text, target_text))

    if not pairs:
        print("⚠️ No matching sentence pairs found.")
        sys.exit(1)

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target'])
        for pair in pairs:
            writer.writerow(pair)

    print(f"✅ Converted {len(pairs)} sentence pairs to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert a TMX file to CSV format.")
    parser.add_argument("tmx_file", type=Path, help="Path to the input TMX file")
    parser.add_argument("csv_file", type=Path, help="Path to output CSV file")
    parser.add_argument("source_lang", type=str, help="Source language code (e.g. 'en')")
    parser.add_argument("target_lang", type=str, help="Target language code (e.g. 'ug')")
    
    args = parser.parse_args()

    if not args.tmx_file.exists():
        print(f"❌ TMX file not found: {args.tmx_file}")
        sys.exit(1)

    tmx_to_csv(
        str(args.tmx_file),
        str(args.csv_file),
        args.source_lang,
        args.target_lang
    )

if __name__ == "__main__":
    main()
