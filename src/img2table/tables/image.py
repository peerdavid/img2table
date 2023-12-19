# coding: utf-8
import copy
from dataclasses import dataclass
from functools import cached_property
from typing import List
from PIL import Image, ImageDraw

import cv2
import numpy as np

from img2table.tables.metrics import compute_img_metrics
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.cells import get_cells
from img2table.tables.processing.bordered_tables.lines import detect_lines, threshold_dark_areas
from img2table.tables.processing.bordered_tables.tables import get_tables
from img2table.tables.processing.bordered_tables.tables.implicit_rows import handle_implicit_rows
from img2table.tables.processing.borderless_tables import identify_borderless_tables
from img2table.tables.processing.prepare_image import prepare_image

@dataclass
class TableImage:
    img: np.ndarray
    min_confidence: int = 50
    char_length: float = None
    median_line_sep: float = None
    thresh: np.ndarray = None
    contours: List[Cell] = None
    lines: List[Line] = None
    tables: List[Table] = None

    def __post_init__(self):
        # Prepare image by removing eventual black background
        self.img = prepare_image(img=self.img)

        # Compute image metrics
        self.char_length, self.median_line_sep, self.contours = compute_img_metrics(img=self.img)

    @cached_property
    def white_img(self) -> np.ndarray:
        white_img = copy.deepcopy(self.img)

        # Draw white rows on detected rows
        for l in self.lines:
            if l.horizontal:
                cv2.rectangle(white_img, (l.x1 - l.thickness, l.y1), (l.x2 + l.thickness, l.y2), (255, 255, 255),
                              3 * l.thickness)
            elif l.vertical:
                cv2.rectangle(white_img, (l.x1, l.y1 - l.thickness), (l.x2, l.y2 + l.thickness), (255, 255, 255),
                              2 * l.thickness)

        return white_img

    def extract_bordered_tables(self, implicit_rows: bool = True, num_add_rows=2, extended_heuristic=True):
        """
        Identify and extract bordered tables from image
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param num_add_rows: number of additional rows to add to each table in case the (multirow /col ) header has no lines
        :return:
        """
        # Apply thresholding
        self.thresh = threshold_dark_areas(img=self.img, char_length=self.char_length)

        # Compute parameters for line detection
        minLinLength = max(int(round(0.33 * self.median_line_sep)), 1) if self.median_line_sep else 10
        maxLineGap = minLinLength / 2
        kernel_size = max(int(round(0.66 * self.median_line_sep)), 1) if self.median_line_sep else 20

        # Detect rows in image
        h_lines, v_lines = detect_lines(thresh=self.thresh,
                                        contours=self.contours,
                                        char_length=self.char_length,
                                        rho=0.3,
                                        theta=np.pi / 180,
                                        threshold=10,
                                        minLinLength=minLinLength,
                                        maxLineGap=maxLineGap,
                                        kernel_size=kernel_size)
        
        if extended_heuristic:
            # Create raw tables without the header in case there is no top line
            # Create cells from rows
            cells = get_cells(horizontal_lines=h_lines,
                            vertical_lines=v_lines)

            # Create tables from rows
            self.tables = get_tables(cells=cells,
                                    elements=self.contours,
                                    lines=h_lines + v_lines,
                                    char_length=self.char_length)

            def not_in_any_table(pos_y):
                for table in self.tables:
                    if table.y1 <= pos_y <= table.y2:
                        return True
                return False

            def check_if_x_in_table(pos_x):
                for table in self.tables:
                    if table.x1 <= pos_x <= table.x2:
                        return True
                return False

            # Now we artifically add new h_lines at the top of each table in case there is a header text wihtout a header line
            # Will be filtered anyway if its not part of the table but this way we ensure there is always a header line.
            # Its based on the average gap a table has between rows / columns
            # h_lines, v_lines = [], []
            for table in self.tables:
                row_ys = [row.y1 for row in table.items]
                col_xs = [col.x1 for col in table.items[0].items]

                if len(row_ys) < 2 or len(col_xs) < 2:
                    continue
 
                gap_h = int(sum([row_ys[i+1] - row_ys[i] for i in range(len(row_ys)-1)]) / (len(row_ys)-1))
                gap_w = int(sum([col_xs[i+1] - col_xs[i] for i in range(len(col_xs)-1)]) / (len(col_xs)-1))

                # Draw some lines around our table to detect non-bordered columns or rows.
                # Note that we keep at least a distance of k to our table to ensure that text cells are still contained.
                k = 5
                num_add_rows += 1
                for i in range(0, num_add_rows):
                    pos = table.y1 - gap_h*i - k
                    if not not_in_any_table(pos):
                        h_lines.append(Line(x1=table.x1, y1=pos, x2=table.x2, y2=pos, thickness=1))
                    
                    pos = table.y2 + gap_h*i + k
                    if not not_in_any_table(pos):
                        h_lines.append(Line(x1=table.x1, y1=pos, x2=table.x2, y2=pos, thickness=1))

                    pos = table.x1 - gap_w*i - k
                    if not check_if_x_in_table(pos):
                        v_lines.append(Line(x1=pos, y1=table.y1, x2=pos, y2=table.y2, thickness=1))
                    
                    pos = table.x2 + gap_w*i + k
                    if not check_if_x_in_table(pos):
                        v_lines.append(Line(x1=pos, y1=table.y1, x2=pos, y2=table.y2, thickness=1))
                

        # Finally, set all our lines
        self.lines = h_lines + v_lines

        # Create cells from rows
        cells = get_cells(horizontal_lines=h_lines,
                          vertical_lines=v_lines)

        # Create tables from rows
        self.tables = get_tables(cells=cells,
                                 elements=self.contours,
                                 lines=self.lines,
                                 char_length=self.char_length)
        

        # If necessary, detect implicit rows
        if implicit_rows:
            self.tables = handle_implicit_rows(img=self.white_img,
                                               tables=self.tables,
                                               contours=self.contours)

        self.tables = [tb for tb in self.tables if tb.nb_rows * tb.nb_columns >= 2]

    def extract_borderless_tables(self):
        """
        Identify and extract borderless tables from image
        :return:
        """
        # Median line separation needs to be not null to extract borderless tables
        if self.median_line_sep is not None:
            # Extract borderless tables
            borderless_tbs = identify_borderless_tables(thresh=self.thresh,
                                                        char_length=self.char_length,
                                                        median_line_sep=self.median_line_sep,
                                                        lines=self.lines,
                                                        contours=self.contours,
                                                        existing_tables=self.tables)

            # Add to tables
            self.tables += [tb for tb in borderless_tbs if tb.nb_rows >= 2 and tb.nb_columns >= 2]

    def extract_tables(self, implicit_rows: bool = False, borderless_tables: bool = False, extended_heuristic: bool = True) -> List[Table]:
        """
        Identify and extract tables from image
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :return: list of identified tables
        """
        # Extract bordered tables
        self.extract_bordered_tables(implicit_rows=implicit_rows, extended_heuristic=extended_heuristic)

        if borderless_tables:
            # Extract borderless tables
            self.extract_borderless_tables()

        return self.tables, self.lines
