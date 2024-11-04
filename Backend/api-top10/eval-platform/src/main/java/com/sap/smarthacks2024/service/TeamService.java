package com.sap.smarthacks2024.service;

import java.util.Optional;
import java.util.UUID;

import com.sap.smarthacks2024.model.Team;

public interface TeamService {
	Optional<Team> getTeamByApiKey(UUID apiKey);
}
 