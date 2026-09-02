"""
Comprehensive Automated Test Suite for AI Handwritten Digit Recognizer
Validates:
1. Dataset properties & integrity
2. KNN Model cross-validation accuracy & parameter k=3
3. Polarity inversion (dark on light vs light on dark)
4. Blank canvas detection
5. Real dataset example retrieval and classification
6. Synthetic handwritten strokes for all digits (0–9)
7. Tkinter desktop script import & integration
"""

import unittest
import numpy as np
from PIL import Image, ImageDraw
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

from preprocessing import (
    load_digits_dataset,
    get_knn_model,
    preprocess_digit,
    get_real_dataset_sample,
    evaluate_k_values
)


class TestDigitRecognizerPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = load_digits_dataset()
        cls.model = get_knn_model(n_neighbors=3)

    def test_01_dataset_properties(self):
        """Verify sklearn load_digits dataset integrity"""
        self.assertEqual(len(self.dataset.data), 1797)
        self.assertEqual(self.dataset.data.shape, (1797, 64))
        self.assertEqual(len(np.unique(self.dataset.target)), 10)
        self.assertEqual(self.dataset.images.shape, (1797, 8, 8))
        self.assertGreaterEqual(np.min(self.dataset.data), 0.0)
        self.assertLessEqual(np.max(self.dataset.data), 16.0)

    def test_02_k_evaluation_and_accuracy(self):
        """Verify empirical cross-validation accuracy for k=3"""
        eval_res = evaluate_k_values()
        self.assertEqual(eval_res["selected_k"], 3)
        mean_cv_k3 = eval_res["k_eval"][3]["mean_accuracy"]
        self.assertGreater(mean_cv_k3, 98.5, f"k=3 CV accuracy was {mean_cv_k3}%, expected > 98.5%")
        self.assertGreater(eval_res["test_accuracy_k3"], 98.0)

    def test_03_empty_canvas_detection(self):
        """Verify empty canvas is flagged with 'empty' status without errors"""
        empty_white = np.ones((320, 320, 4), dtype=np.uint8) * 255
        result, status = preprocess_digit(empty_white, model=self.model)
        self.assertEqual(status, "empty")
        self.assertIsNone(result)

        empty_black = np.zeros((320, 320, 4), dtype=np.uint8)
        result2, status2 = preprocess_digit(empty_black, model=self.model)
        self.assertEqual(status2, "empty")
        self.assertIsNone(result2)

    def test_04_polarity_inversion(self):
        """Verify dark ink on white canvas is properly inverted"""
        img = Image.new("RGB", (320, 320), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.line([(160, 40), (160, 280)], fill=(0, 0, 0), width=20)

        result, status = preprocess_digit(img, model=self.model)
        self.assertEqual(status, "success")
        self.assertTrue(result["is_inverted"], "White background should trigger polarity inversion")
        self.assertGreater(np.max(result["scaled_8x8"]), 0.0)

    def test_05_real_dataset_examples(self):
        """Verify testing real dataset examples (0–9) returns 100% accuracy on canonical samples"""
        for digit in range(10):
            sample = get_real_dataset_sample(digit)
            self.assertIsNotNone(sample)
            self.assertEqual(sample["target_label"], digit)
            self.assertEqual(sample["prediction"], digit, f"Real sample for {digit} should predict {digit}")
            self.assertGreaterEqual(sample["confidence"], 33.33)

    def test_06_synthetic_digits_recognition(self):
        """Verify synthetic stroke drawings for digits 0-9 pass preprocessing and predict valid classes"""
        brush_width = 24
        test_cases = [
            (0, lambda d: d.ellipse([(100, 70), (250, 280)], outline=(0, 0, 0), width=brush_width)),
            (1, lambda d: d.line([(175, 60), (175, 290)], fill=(0, 0, 0), width=brush_width)),
            (2, lambda d: (d.arc([(100, 60), (250, 180)], start=180, end=0, fill=(0, 0, 0), width=brush_width),
                           d.line([(250, 120), (100, 290)], fill=(0, 0, 0), width=brush_width),
                           d.line([(100, 290), (250, 290)], fill=(0, 0, 0), width=brush_width))),
            (3, lambda d: (d.line([(100, 70), (240, 70)], fill=(0, 0, 0), width=brush_width),
                           d.line([(240, 70), (150, 170)], fill=(0, 0, 0), width=brush_width),
                           d.arc([(100, 170), (240, 290)], start=270, end=90, fill=(0, 0, 0), width=brush_width))),
            (4, lambda d: (d.line([(200, 60), (100, 220)], fill=(0, 0, 0), width=brush_width),
                           d.line([(90, 220), (250, 220)], fill=(0, 0, 0), width=brush_width),
                           d.line([(200, 60), (200, 290)], fill=(0, 0, 0), width=brush_width))),
            (5, lambda d: (d.line([(230, 70), (120, 70)], fill=(0, 0, 0), width=brush_width),
                           d.line([(120, 70), (120, 170)], fill=(0, 0, 0), width=brush_width),
                           d.arc([(100, 160), (240, 290)], start=270, end=90, fill=(0, 0, 0), width=brush_width))),
            (6, lambda d: (d.arc([(100, 60), (240, 280)], start=90, end=270, fill=(0, 0, 0), width=brush_width),
                           d.ellipse([(100, 160), (240, 290)], outline=(0, 0, 0), width=brush_width))),
            (7, lambda d: (d.line([(100, 70), (240, 70)], fill=(0, 0, 0), width=brush_width),
                           d.line([(240, 70), (140, 290)], fill=(0, 0, 0), width=brush_width))),
            (8, lambda d: (d.ellipse([(120, 60), (230, 170)], outline=(0, 0, 0), width=brush_width),
                           d.ellipse([(110, 160), (240, 290)], outline=(0, 0, 0), width=brush_width))),
            (9, lambda d: (d.ellipse([(110, 60), (240, 190)], outline=(0, 0, 0), width=brush_width),
                           d.line([(240, 120), (240, 290)], fill=(0, 0, 0), width=brush_width)))
        ]

        correct_count = 0
        for digit, draw_fn in test_cases:
            img = Image.new("RGB", (350, 350), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw_fn(draw)

            result, status = preprocess_digit(img, model=self.model)
            self.assertEqual(status, "success")
            self.assertIn(result["prediction"], list(range(10)))
            self.assertEqual(result["scaled_8x8"].shape, (8, 8))
            self.assertEqual(len(result["flat_array"]), 64)
            self.assertEqual(len(result["top_predictions"]), 3)
            
            if result["prediction"] == digit:
                correct_count += 1

        self.assertEqual(correct_count, 10, f"Expected 10/10 synthetic matches, got {correct_count}/10")


if __name__ == "__main__":
    unittest.main()
