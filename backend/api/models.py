
from django.db import models
from django.contrib.auth.models import User

class Prediction(models.Model):
    user             = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    article_text     = models.TextField()
    source_url       = models.URLField(blank=True, null=True)
    label            = models.CharField(max_length=10)
    confidence       = models.FloatField()
    shap_highlights  = models.JSONField(default=list)
    predicted_at     = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-predicted_at']

    def __str__(self):
        return f'{self.label} ({self.confidence:.0%})'
