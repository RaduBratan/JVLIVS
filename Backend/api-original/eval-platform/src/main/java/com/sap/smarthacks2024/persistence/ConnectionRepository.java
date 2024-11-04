package com.sap.smarthacks2024.persistence;

import java.util.List;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import com.sap.smarthacks2024.model.Connection;
import com.sap.smarthacks2024.model.ConnectionOverCapacityView;

public interface ConnectionRepository extends JpaRepository<Connection, UUID> {

	@Query("""
			         SELECT
			           m.connection as connection,
			           sum(m.amount) as amountInTransit
			         FROM Movement m
			         WHERE
			           m.session.id = :sessionId
			           AND m.dayPosted <= :day
			           AND m.dayDelivered >= :day
			         GROUP By m.connection, m.connection.from.id, m.connection.to.id, m.connection.leadTimeDays, m.connection.connectionType, m.connection.maxCapacity
			         HAVING sum(m.amount) > m.connection.maxCapacity
			""")
	List<ConnectionOverCapacityView> findConnectionsOverCapacity(UUID sessionId, int day);
}
