from django.db import transaction
from django.utils import timezone
from django.shortcuts import get_object_or_404
from rest_framework import serializers

from attachments.models import Attachment
from attachments.serializers import SimpleAttachmentSerializer
from events.models import Event
from organizations.serializers import SimpleOrganizationSerializer
from transactions.models import Transaction
from transactions.constants import TransactionStatus, TransactionType
from django.db.models import Sum

class EventSerializer(serializers.ModelSerializer):
    organization = SimpleOrganizationSerializer(read_only=True)
    attachments = SimpleAttachmentSerializer(read_only=True, many=True)
    current_amount = serializers.SerializerMethodField(read_only=True)
    
    uploaded_attachments = serializers.ListField(
        child=serializers.FileField(),
        write_only=True,
        required=False
    )

    class Meta:
        model = Event
        fields = [
            "id",
            "organization",
            "title",
            "status",
            "description",
            "target_amount",
            "current_amount",
            "attachments",
            "start_date",
            "end_date",
            "uploaded_attachments",
        ]
        read_only_fields = ["start_date"]

    def validate(self, attrs):
        organization = self.context.get("organization")

        if attrs["end_date"] <= timezone.now():
            raise serializers.ValidationError("End date must be in the future.")

        if organization.kpay_qr_url is None or organization.phone_number is None:
            raise serializers.ValidationError(
                "Organization is not set up properly yet to create events."
            )

        return super().validate(attrs)

    def create(self, validated_data):
        attachments = validated_data.pop("uploaded_attachments", [])
        with transaction.atomic():
            event = Event.objects.create(**validated_data)
            for attachment in attachments:
                Attachment.objects.create(content_object=event, file=attachment)

            return event

    def get_current_amount(self, obj):
        event = get_object_or_404(Event, pk=obj.id)    
        result = Transaction.objects.filter(
            event=event,
            type=TransactionType.DONATION, 
            status=TransactionStatus.APPROVED
        ).aggregate(total=Sum("amount"))
        return result['total'] if result['total'] is not None else 0

class SimpleEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = Event
        fields = [
            "id",
            "organization",
            "title",
            "status",
        ]
