package com.sap.smarthacks2024.persistence;

import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

import com.sap.smarthacks2024.model.Team;

public interface TeamRepository extends JpaRepository<Team, UUID> {
	Optional<Team> findByApiKey(UUID apiKey);
}
