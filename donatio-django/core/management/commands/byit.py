import json
import os
import random
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.core.files import File
from django.core.management.base import BaseCommand
from django.utils import timezone

from accounts.models import User
from activities.models import Activity, ActivityTransaction
from attachments.models import Attachment
from events.constants import EventStatusChoices
from events.models import Event
from organizations.constants import OrganizationRequestStatus
from organizations.models import Organization, OrganizationRequest
from transactions.constants import TransactionStatus, TransactionType
from transactions.models import Transaction


class Command(BaseCommand):
    help = "Seed dummy org requests, orgs, transactions, activities, and attachments."

    def handle(self, *args, **kwargs):
        ### Load Data ###
        json_file = os.path.join(os.getcwd(), "core", "data", "seed_data.json")
        with open(json_file, "r") as file:
            data = json.load(file)

        self.stdout.write(self.style.NOTICE("✅ Starting dummy data seeding..."))

        ### Get System Admin ###
        approver = User.objects.filter(is_superuser=True).first()

        for entry in data:
            username = entry["admin"]
            admin = User.objects.filter(username=username).first()
            if not admin:
                self.stdout.write(
                    self.style.WARNING(f"⚠️ Skipping: user '{username}' not found")
                )
                continue

            org_name = entry["organization_name"]
            org_type = entry["organization_type"]

            ### Create Organization Request ###
            org_request = OrganizationRequest.objects.create(
                submitted_by=admin,
                organization_name=org_name,
                type=org_type,
                status=OrganizationRequestStatus.APPROVED,
                approved_by=approver,
                approved_at=timezone.now(),
            )

            org = Organization.objects.create(
                admin=admin,
                name=org_name,
                type=org_type,
                description=entry["organization_description"],
                phone_number=entry["organization_phone"],
                email=entry["organization_email"],
                organization_request=org_request,
            )

            self.create_attachments(
                org, os.path.join(os.getcwd(), "core", "data", "example.png")
            )

            ### Create Donations ###
            for donation in entry.get("donations", []):
                tx = Transaction.objects.create(
                    organization=org,
                    actor=admin,
                    title=donation["title"],
                    amount=donation["amount"],
                    status=TransactionStatus.APPROVED,
                    type=TransactionType.DONATION,
                )

            ### Create Disbursement ###
            disbursements = []
            for disbursement in entry.get("disbursements", []):
                tx = Transaction.objects.create(
                    organization=org,
                    actor=admin,
                    title=donation["title"],
                    amount=donation["amount"],
                    status=TransactionStatus.APPROVED,
                    type=TransactionType.DISBURSEMENT,
                )
                disbursements.append(disbursement)

            ### Create Activity ###
            for activity in entry.get("activities", []):
                act = Activity.objects.create(
                    organization=org,
                    title=activity["title"],
                    description=activity["description"],
                    location=activity["location"],
                )

                for tx in disbursements:
                    ActivityTransaction.objects.create(activity=act, transaction=tx)

            ### Create Event ###
            for event in entry.get("events", []):
                evt = Event.objects.create(
                    organization=org,
                    title=event["title"],
                    description=event["description"],
                    target_amount=event["target_amount"],
                    status=EventStatusChoices.OPEN,
                    start_date=timezone.now(),
                    end_date=timezone.now() + timedelta(days=10),
                )

            self.stdout.write(
                self.style.SUCCESS(f"✅ Seeded data for org '{org_name}'")
            )

        self.stdout.write(self.style.SUCCESS("🎉 All data seeded from JSON!"))

    def create_attachments(self, instance, file_path, count=1, max_allowed=None):
        for _ in range(min(count, max_allowed or count)):
            with open(file_path, "rb") as f:
                Attachment.objects.create(
                    content_object=instance, file=File(f, name=f"default_{uuid4()}.jpg")
                )
