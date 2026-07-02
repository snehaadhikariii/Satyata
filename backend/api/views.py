from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Prediction
from .serializers import PredictionSerializer, AnalyzeInputSerializer
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ml.predict import predict as ml_predict

class AnalyzeView(APIView):
    def post(self, request):
        serializer = AnalyzeInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)
        text = serializer.validated_data.get('text', '')
        url  = serializer.validated_data.get('url', '')
        try:
            result = ml_predict(text=text or None, url=url or None)
        except ValueError as e:
            return Response({'error': str(e)}, status=400)
        except Exception as e:
            return Response({'error': f'Prediction failed: {str(e)}'}, status=500)
        pred = Prediction.objects.create(
            article_text    = result['article_text'],
            source_url      = url or None,
            label           = result['label'],
            confidence      = result['confidence'],
            shap_highlights = result['highlights'],
        )
        return Response({'id': pred.id, **result})

class HistoryView(APIView):
    def get(self, request):
        preds = Prediction.objects.all()[:50]
        return Response(PredictionSerializer(preds, many=True).data)

class HealthView(APIView):
    def get(self, request):
        return Response({'status': 'ok', 'message': 'Satyata API is running'})