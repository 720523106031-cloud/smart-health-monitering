"""
Automated Test Suite for Smart Health Monitoring System
Validates Flask routes, ML model predictions, clinical threshold validations,
hypertensive crisis alerts, and CSV export functionality.
"""

import unittest
import json
import os
import sys

# Ensure root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api.index import app

class SmartHealthMonitoringTests(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    # ---------------------------------------------------------
    # 1. HTML Web Page Routes Tests
    # ---------------------------------------------------------
    def test_home_page(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'SMART HEALTH MONITORING SYSTEM', res.data)

    def test_sugar_page(self):
        res = self.client.get('/sugar')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Blood Sugar Monitoring', res.data)

    def test_blood_pressure_page(self):
        res = self.client.get('/blood-pressure')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Blood Pressure', res.data)

    def test_dashboard_page(self):
        res = self.client.get('/dashboard')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Health Monitoring Dashboard', res.data)

    def test_history_page(self):
        res = self.client.get('/history')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Reading History', res.data)

    def test_about_page(self):
        res = self.client.get('/about')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'DevOps Architecture', res.data)

    def test_disclaimer_page(self):
        res = self.client.get('/disclaimer')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Medical Disclaimer', res.data)

    def test_contact_page(self):
        res = self.client.get('/contact')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Contact & Developer Support', res.data)

    # ---------------------------------------------------------
    # 2. Blood Sugar Prediction API Tests
    # ---------------------------------------------------------
    def test_predict_sugar_normal(self):
        payload = {
            'age': 45,
            'gender': 'Male',
            'glucose_level': 92.0,
            'measurement_type': 'Fasting'
        }
        res = self.client.post('/api/predict/sugar',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn('Normal', data['data']['clinical_category'])
        self.assertFalse(data['data']['is_hypo'])

    def test_predict_sugar_prediabetes(self):
        payload = {
            'age': 52,
            'gender': 'Female',
            'glucose_level': 118.0,
            'measurement_type': 'Fasting'
        }
        res = self.client.post('/api/predict/sugar',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn('Prediabetes', data['data']['clinical_category'])

    def test_predict_sugar_hypoglycemia(self):
        payload = {
            'age': 30,
            'gender': 'Male',
            'glucose_level': 58.0,
            'measurement_type': 'Fasting'
        }
        res = self.client.post('/api/predict/sugar',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertTrue(data['data']['is_hypo'])
        self.assertIn('Hypoglycemia', data['data']['clinical_category'])

    def test_predict_sugar_invalid_range(self):
        payload = {
            'age': 45,
            'gender': 'Male',
            'glucose_level': 850.0,  # Out of range
            'measurement_type': 'Fasting'
        }
        res = self.client.post('/api/predict/sugar',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 400)

    # ---------------------------------------------------------
    # 3. Blood Pressure Prediction API Tests
    # ---------------------------------------------------------
    def test_predict_bp_normal(self):
        payload = {
            'age': 40,
            'gender': 'Male',
            'systolic': 115,
            'diastolic': 75,
            'pulse_rate': 72
        }
        res = self.client.post('/api/predict/bp',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn('Normal', data['data']['clinical_category'])
        self.assertFalse(data['data']['is_crisis'])

    def test_predict_bp_hypertensive_crisis(self):
        # Systolic > 180 or Diastolic > 120 must trigger crisis safety alert
        payload = {
            'age': 62,
            'gender': 'Male',
            'systolic': 192,
            'diastolic': 125,
            'pulse_rate': 94
        }
        res = self.client.post('/api/predict/bp',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertTrue(data['data']['is_crisis'])
        self.assertIn('Crisis', data['data']['clinical_category'])

    def test_predict_bp_invalid_inputs(self):
        payload = {
            'age': 50,
            'gender': 'Male',
            'systolic': 400,  # Invalid
            'diastolic': 80,
            'pulse_rate': 72
        }
        res = self.client.post('/api/predict/bp',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 400)

    # ---------------------------------------------------------
    # 4. Comprehensive Dual Prediction API Tests
    # ---------------------------------------------------------
    def test_predict_comprehensive_normal(self):
        payload = {
            'age': 35,
            'gender': 'Female',
            'glucose_level': 95.0,
            'measurement_type': 'Fasting',
            'systolic': 118,
            'diastolic': 78,
            'pulse_rate': 72
        }
        res = self.client.post('/api/predict/comprehensive',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn('sugar', data['data'])
        self.assertIn('bp', data['data'])
        self.assertIn('Normal', data['data']['sugar']['clinical_category'])
        self.assertIn('Normal', data['data']['bp']['clinical_category'])

    def test_predict_comprehensive_crisis(self):
        payload = {
            'age': 65,
            'gender': 'Male',
            'glucose_level': 240.0,
            'measurement_type': 'Post-meal',
            'systolic': 190,
            'diastolic': 125,
            'pulse_rate': 98
        }
        res = self.client.post('/api/predict/comprehensive',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertTrue(data['data']['bp']['is_crisis'])
        self.assertIn('Crisis', data['data']['bp']['clinical_category'])

    def test_contact_api_endpoint(self):
        payload = {
            'name': 'Professor Johnson',
            'email': 'prof.johnson@university.edu',
            'subject': 'Academic Evaluation',
            'message': 'Excellent DevOps pipeline and healthcare risk modeling assignment.'
        }
        res = self.client.post('/api/contact',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])

    # ---------------------------------------------------------
    # 4. Model Metadata & Evaluation API Tests
    # ---------------------------------------------------------
    def test_model_metrics_api(self):
        res = self.client.get('/api/model-metrics')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn('glucose_model', data['metadata'])
        self.assertIn('bp_model', data['metadata'])
        self.assertGreater(data['metadata']['glucose_model']['accuracy'], 0.90)
        self.assertGreater(data['metadata']['bp_model']['accuracy'], 0.90)

    # ---------------------------------------------------------
    # 5. CSV Export API Tests
    # ---------------------------------------------------------
    def test_csv_export_endpoint(self):
        payload = {
            'readings': [
                {
                    'id': 'GLU-TEST-1',
                    'timestamp': '2026-11-12 10:00:00',
                    'type': 'sugar',
                    'age': 45,
                    'gender': 'Male',
                    'glucose_level': 98.0,
                    'measurement_type': 'Fasting',
                    'ml_risk': 'Normal / Low Risk',
                    'clinical_category': 'Normal Fasting Glucose'
                },
                {
                    'id': 'BP-TEST-1',
                    'timestamp': '2026-11-12 10:05:00',
                    'type': 'bp',
                    'age': 45,
                    'gender': 'Male',
                    'systolic': 120,
                    'diastolic': 80,
                    'pulse_rate': 70,
                    'ml_risk': 'Elevated / Stage 1 Hypertension',
                    'clinical_category': 'Hypertension Stage 1'
                }
            ]
        }
        res = self.client.post('/api/export/csv',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'text/csv')
        self.assertIn(b'GLU-TEST-1', res.data)
        self.assertIn(b'BP-TEST-1', res.data)

if __name__ == '__main__':
    unittest.main()

