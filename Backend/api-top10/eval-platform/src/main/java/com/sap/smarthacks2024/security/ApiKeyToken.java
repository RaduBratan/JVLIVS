package com.sap.smarthacks2024.security;

import java.util.Collection;

import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;

public class ApiKeyToken implements Authentication {

	private static final long serialVersionUID = 1L;

	private final String apiKey;
	private final String teamId;
	private final String teamName;

	private boolean authenticated;

	public ApiKeyToken(String apiKey, String teamId, String teamName) {
		this.apiKey = apiKey;
		this.teamId = teamId;
		this.teamName = teamName;
	}

	public ApiKeyToken() {
		this.apiKey = null;
		this.teamId = null;
		this.teamName = null;

	}

	@Override
	public String getName() {
		return teamId;
	}

	@Override
	public Collection<? extends GrantedAuthority> getAuthorities() {
		return null;
	}

	@Override
	public Object getCredentials() {
		return apiKey;
	}

	@Override
	public Object getDetails() {
		return teamName;
	}

	@Override
	public Object getPrincipal() {
		return teamId;
	}

	@Override
	public boolean isAuthenticated() {
		return authenticated;
	}

	@Override
	public void setAuthenticated(boolean isAuthenticated) throws IllegalArgumentException {
		this.authenticated = isAuthenticated;
	}

}
