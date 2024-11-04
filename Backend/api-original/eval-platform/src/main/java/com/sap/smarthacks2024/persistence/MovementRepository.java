package com.sap.smarthacks2024.persistence;

import java.util.List;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import com.sap.smarthacks2024.model.Movement;

public interface MovementRepository extends JpaRepository<Movement, UUID> {

	@Query("SELECT m FROM Movement m WHERE m.session.id = :sessionId AND m.dayDelivered = :currentDay")
	List<Movement> findBySessionIdAndDayDelivered(UUID sessionId, int currentDay);

	@Query("SELECT m FROM Movement m WHERE m.session.id = :sessionId AND m.dayPosted = :currentDay")
	List<Movement> findBySessionIdAndDayPosted(UUID sessionId, int currentDay);
	
	@Query("SELECT m from Movement m where m.session.id = :sessionId AND m.dayPosted <= :currentDay AND m.dayDelivered >= :currentDay")
	List<Movement> findBySessionIdAndInTransit(UUID sessionId, int currentDay);

}
