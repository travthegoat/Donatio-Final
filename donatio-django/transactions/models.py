from django.conf import settings
from django.core.validators import MinValueValidator
from django.db import models

from core.models import AttachableModel, BaseModel
from events.models import Event
from organizations.models import Organization

from .constants import TransactionStatus, TransactionType


class Transaction(BaseModel, AttachableModel):
    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, related_name="transactions"
    )
    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="transactions"
    )
    event = models.ForeignKey(
        Event,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="transactions",
    )
    title = models.CharField(max_length=100, null=True, blank=True)
    amount = models.DecimalField(
        max_digits=10, decimal_places=2, validators=[MinValueValidator(0)]
    )
    type = models.CharField(max_length=20, choices=TransactionType.choices)
    status = models.CharField(
        max_length=20,
        choices=TransactionStatus.choices,
        default=TransactionStatus.PENDING,
    )
    review_required = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Transaction"
        verbose_name_plural = "Transactions"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.actor.username} {self.amount} {self.type} {self.status} {self.created_at}"
