from rest_framework import serializers
from .models import Prediction

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['id', 'article_text', 'source_url', 'label',
                  'confidence', 'shap_highlights', 'predicted_at']
        read_only_fields = ['label', 'confidence', 'shap_highlights', 'predicted_at']

class AnalyzeInputSerializer(serializers.Serializer):
    text = serializers.CharField(required=False, allow_blank=True)
    url  = serializers.URLField(required=False, allow_blank=True)

    def validate(self, data):
        if not data.get('text') and not data.get('url'):
            raise serializers.ValidationError('Provide either text or url.')
        return data