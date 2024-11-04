package com.sap.smarthacks2024.security;

import java.util.UUID;

import org.springframework.stereotype.Component;

import com.sap.smarthacks2024.service.TeamService;

@Component
public class ApiKeyStoreImpl implements ApiKeyStore {

	private final TeamService teamService;

	ApiKeyStoreImpl(TeamService teamService) {
		this.teamService = teamService;
	}

	@Override
	public ApiKeyToken getApiKeyToken(String apiKey) {
		UUID apiKeyUUID = UUID.fromString(apiKey);
		return teamService.getTeamByApiKey(apiKeyUUID)
				.map(team -> new ApiKeyToken(team.getApiKey().toString(), team.getId().toString(), team.getName()))
				.orElse(new ApiKeyToken());
	}

}
