package com.sap.smarthacks2024.service.impl;

import java.util.Optional;
import java.util.UUID;

import org.springframework.stereotype.Service;

import com.sap.smarthacks2024.model.Team;
import com.sap.smarthacks2024.persistence.TeamRepository;
import com.sap.smarthacks2024.service.TeamService;

@Service
public class TeamServiceImpl implements TeamService {
	
	private final TeamRepository teamRepository;
	
	TeamServiceImpl(TeamRepository teamRepository) {
        this.teamRepository = teamRepository;
	}

	@Override
	public Optional<Team> getTeamByApiKey(UUID apiKey) {
		return teamRepository.findByApiKey(apiKey);
	}

}
