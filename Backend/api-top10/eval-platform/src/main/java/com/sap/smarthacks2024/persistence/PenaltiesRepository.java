package com.sap.smarthacks2024.persistence;

import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

import com.sap.smarthacks2024.model.Penalty;

public interface PenaltiesRepository extends JpaRepository<Penalty, UUID> {

}
