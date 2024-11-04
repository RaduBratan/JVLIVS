package com.sap.smarthacks2024.security;

import org.springframework.security.core.Authentication;
import org.springframework.security.web.server.authentication.ServerAuthenticationConverter;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;

import reactor.core.publisher.Mono;

@Component
public class ApiKeyAuthenticationConverter implements ServerAuthenticationConverter {
	
	private final ApiKeyStore apiKeyStore;
	
	public ApiKeyAuthenticationConverter(ApiKeyStore apiKeyStore) {
		this.apiKeyStore = apiKeyStore;
	}

	@Override
	public Mono<Authentication> convert(ServerWebExchange exchange) {
		return Mono.justOrEmpty(exchange)
		 .flatMap(exch-> Mono.justOrEmpty(exch.getRequest().getHeaders().get("API-KEY")))
		 .filter(headers-> !headers.isEmpty())
		 .map(headers-> headers.get(0))
		 .map(apiKeyStore::getApiKeyToken);
	}

}
