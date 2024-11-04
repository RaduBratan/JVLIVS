package com.sap.smarthacks2024.security;

import org.springframework.security.authentication.ReactiveAuthenticationManager;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Component;

import io.micrometer.common.util.StringUtils;
import reactor.core.publisher.Mono;

@Component
public class ApiKeyAuthenticationManager implements ReactiveAuthenticationManager {

	@Override
	public Mono<Authentication> authenticate(Authentication authentication) {
		if (authentication != null && authentication.getCredentials() != null) {
			authentication.setAuthenticated(StringUtils.isNotBlank(authentication.getCredentials().toString()));
		}
		return Mono.just(authentication);
	}

}
