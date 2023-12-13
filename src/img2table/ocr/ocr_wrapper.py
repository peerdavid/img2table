# coding: utf-8

from typing import List, Tuple, Dict

import polars as pl

from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe

supported = True
try:
    from ocr_wrapper import BBox
except ImportError:
    supported = False

class OCRWrapper(OCRInstance):
    """
    ocr_wrapper instance
    """
    def __init__(self):
        if not supported:
            raise ImportError("OCRWrapper is not supported on this platform. Please install ocr_wrapper first.")

    def content(self, document: Document) -> List[List[Tuple]]:
        if document.bboxes is None:
            raise Exception("No bboxes found for this document")

        return document.bboxes

    def to_ocr_dataframe(self, content: List[List[dict]]) -> OCRDataframe:
        """
        Convert hOCR HTML to OCRDataframe object
        :param content: hOCR HTML string
        :return: OCRDataframe object corresponding to content
        """

        #Now convert all bboxes which are a list of pages into the dict_word format
        ret = []
        for page, ocr_result in enumerate(content):
            word_id = 0
            for bbox in ocr_result:
                word_id += 1
                w, h = bbox["bbox"].original_width, bbox["bbox"].original_height
                ret.append({
                    "page": page,
                    "class": "ocrx_word",
                    "id": f"word_{page + 1}_{word_id}",
                    "parent": f"word_{page + 1}_{word_id}",
                    "value": bbox["text"],
                    "confidence": round(100 * bbox["confidence"]),
                    "x1": round(bbox["bbox"].TLx * w),
                    "y1": round(bbox["bbox"].TLy * h),
                    "x2": round(bbox["bbox"].BRx * w),
                    "y2": round(bbox["bbox"].BRy * h),
                })

        ret = OCRDataframe(df=pl.LazyFrame(ret)) if ret else None
        return ret
