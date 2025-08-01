from rest_framework import serializers

from accounts.serializers import CustomUserDetailsSerializer
from organizations.serializers import SimpleOrganizationSerializer

from .models import Chat, ChatMessage


class ChatSerializer(serializers.ModelSerializer):
    donor = CustomUserDetailsSerializer(read_only=True)
    organization = SimpleOrganizationSerializer(read_only=True)

    class Meta:
        model = Chat
        fields = "__all__"


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = "__all__"


class UpdateChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ["content"]
