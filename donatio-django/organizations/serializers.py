from django.contrib.auth import get_user_model
from django.db import transaction
from django.db.models import Sum
from rest_framework import serializers, status

from attachments.models import Attachment
from attachments.serializers import SimpleAttachmentSerializer
from transactions.constants import TransactionStatus, TransactionType

from .constants import OrganizationRequestStatus
from .models import Organization, OrganizationRequest
from .utils import extract_qr_url

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email"]


class CreateOrganizationRequestSerializer(serializers.ModelSerializer):
    uploaded_attachments = serializers.ListField(
        child=serializers.FileField(),
        write_only=True,
        required=True,
    )
    attachments = SimpleAttachmentSerializer(many=True, read_only=True)

    class Meta:
        model = OrganizationRequest
        fields = [
            "id",
            "organization_name",
            "type",
            "uploaded_attachments",
            "attachments",
        ]

    def create(self, validated_data):
        attachments_data = validated_data.pop("uploaded_attachments", [])
        with transaction.atomic():
            organization_request = OrganizationRequest.objects.create(**validated_data)

            for file in attachments_data:
                Attachment.objects.create(
                    content_object=organization_request, file=file
                )

        return organization_request


class UpdateOrganizationRequestSerializer(serializers.ModelSerializer):
    attachments = SimpleAttachmentSerializer(many=True, read_only=True)

    class Meta:
        model = OrganizationRequest
        fields = ["id", "organization_name", "type", "attachments", "status"]
        read_only_fields = ["id", "organization_name", "type", "attachments"]

    def validate_status(self, value):
        if self.instance.status == OrganizationRequestStatus.APPROVED:
            raise serializers.ValidationError("Cannot update status after approval")
        return value


class OrganizationRequestSerializer(serializers.ModelSerializer):
    submitted_by = UserSerializer(read_only=True)
    approved_by = UserSerializer(read_only=True)
    attachments = serializers.SerializerMethodField(read_only=True)

    def get_attachments(self, obj):
        return SimpleAttachmentSerializer(obj.attachments.all(), many=True).data

    class Meta:
        model = OrganizationRequest
        fields = [
            "id",
            "organization_name",
            "status",
            "type",
            "submitted_by",
            "approved_by",
            "approved_at",
            "created_at",
            "updated_at",
            "attachments",
        ]


class OrganizationStatsSerializer(serializers.ModelSerializer):
    total_received_money = serializers.SerializerMethodField(read_only=True)
    total_expense = serializers.SerializerMethodField(read_only=True)
    total_current_balance = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Organization
        fields = ["total_received_money", "total_expense", "total_current_balance"]

    def get_total_received_money(self, obj):
        return (
            obj.transactions.filter(type=TransactionType.DONATION).aggregate(
                total=Sum("amount")
            )["total"]
            or 0
        )

    def get_total_expense(self, obj):
        return (
            obj.transactions.filter(type=TransactionType.DISBURSEMENT).aggregate(
                total=Sum("amount")
            )["total"]
            or 0
        )

    def get_total_current_balance(self, obj):
        total_received = self.get_total_received_money(obj)
        total_expense = self.get_total_expense(obj)
        return total_received - total_expense


class OrganizationSerializer(serializers.ModelSerializer):
    admin = UserSerializer(read_only=True)
    attachments = serializers.SerializerMethodField(read_only=True)
    uploaded_attachments = serializers.ListField(
        child=serializers.FileField(allow_empty_file=True),
        write_only=True,
        required=False,
        allow_empty=True,
    )
    organization_request = OrganizationRequestSerializer(read_only=True)
    stats = serializers.SerializerMethodField(read_only=True)
    total_donations = serializers.SerializerMethodField(read_only=True)
    total_donors = serializers.SerializerMethodField(read_only=True)

    def get_stats(self, obj):
        return OrganizationStatsSerializer(obj).data

    def get_attachments(self, obj):
        return SimpleAttachmentSerializer(obj.attachments.all(), many=True).data

    def get_total_donors(self, obj):
        return (
            obj.transactions.filter(
                type=TransactionType.DONATION, status=TransactionStatus.APPROVED
            )
            .values("actor")
            .distinct()
            .count()
        )

    def get_total_donations(self, obj):
        return obj.transactions.filter(
            type=TransactionType.DONATION, status=TransactionStatus.APPROVED
        ).count()

    class Meta:
        model = Organization
        fields = [
            "id",
            "admin",
            "name",
            "type",
            "kpay_qr_url",
            "description",
            "phone_number",
            "email",
            "organization_request",
            "additional_info",
            "created_at",
            "updated_at",
            "attachments",
            "kpay_qr_image",
            "uploaded_attachments",
            "stats",
            "total_donations",
            "total_donors",
        ]
        read_only_fields = [
            "name",
            "type",
            "created_at",
            "updated_at",
            "stats",
            "kpay_qr_url",
        ]

    def update(self, instance, validated_data):
        attachments_data = validated_data.pop("uploaded_attachments", [])
        qr_code_file = validated_data.get("kpay_qr_image", None)

        with transaction.atomic():
            # Update all other fields on the instance
            for attr, value in validated_data.items():
                setattr(instance, attr, value)

            # If there are new attachments, delete old ones and create new ones
            if attachments_data:
                # Delete old attachments
                instance.attachments.all().delete()
                # Loop for first 2 attachments
                for file in attachments_data[:2]:
                    if file:
                        Attachment.objects.create(content_object=instance, file=file)

            if qr_code_file:
                print("hello")
                qr_code_url = extract_qr_url(qr_code_file)
                print(qr_code_url)
                instance.kpay_qr_url = qr_code_url

            instance.save()

        return instance

    def validate_kpay_qr_url(self, value):
        if not value:
            raise serializers.ValidationError("Kpay QR URL is required")

        return value

    def validate_phone_number(self, value):
        if not value:
            raise serializers.ValidationError("Phone number is required")

        if not value.isdigit():
            raise serializers.ValidationError("Phone number must be digits")

        if value.startswith("09"):
            raise serializers.ValidationError("Phone number must not start with 09")

        if value.startswith("0"):
            raise serializers.ValidationError("Phone number must not start with 0")

        if len(value) < 8 or len(value) > 10:
            raise serializers.ValidationError("Phone number must be 8-10 digits")

        return value

    def create(self, validated_data):
        if "admin" not in validated_data and self.context["request"].user.is_staff:
            validated_data["admin"] = self.context["request"].user
        return super().create(validated_data)


class SimpleOrganizationSerializer(serializers.ModelSerializer):
    attachments = SimpleAttachmentSerializer(many=True, read_only=True)

    class Meta:
        model = Organization
        fields = ["id", "admin", "name", "attachments"]
