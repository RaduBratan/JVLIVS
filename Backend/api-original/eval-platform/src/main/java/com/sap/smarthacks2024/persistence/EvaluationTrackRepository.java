package com.sap.smarthacks2024.persistence;

import java.util.List;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;

import com.sap.smarthacks2024.model.EvaluationTrack;

public interface EvaluationTrackRepository extends JpaRepository<EvaluationTrack, UUID> {

	List<EvaluationTrack> findByProdDay(int day);

	@Modifying
	@Query("UPDATE EvaluationTrack e SET e.latest = true WHERE e.latest = false")
	void markAsNotLatest();
}
