package com.sap.smarthacks2024.persistence;

import java.util.List;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import com.sap.smarthacks2024.model.DemandStatus;

public interface DemandStatusRepository extends JpaRepository<DemandStatus, UUID> {

	@Query("SELECT d FROM DemandStatus d WHERE d.session.id = :sessionId AND d.demand.postDay = :postDay")
	List<DemandStatus> findBySessionIdAndPostDay(UUID sessionId, int postDay);

	
	@Query("SELECT d FROM DemandStatus d WHERE d.session.id = :sessionId AND d.remainingQuantity > 0")
	List<DemandStatus> findBySessionIdAndHavingRemainingQuantities(UUID sessionId);

}
