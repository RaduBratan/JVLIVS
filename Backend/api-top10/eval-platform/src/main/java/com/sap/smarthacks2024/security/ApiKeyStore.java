package com.sap.smarthacks2024.security;

public interface ApiKeyStore {
	ApiKeyToken getApiKeyToken(String apiKey);
}
