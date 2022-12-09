from django.db import models
from django.contrib.postgres.fields import ArrayField

# Create your models here.

class UserConversation(models.Model):
    name = models.CharField(max_length=30)
    message=models.CharField(max_length=30)
    conversation=ArrayField(models.CharField(max_length=100))
