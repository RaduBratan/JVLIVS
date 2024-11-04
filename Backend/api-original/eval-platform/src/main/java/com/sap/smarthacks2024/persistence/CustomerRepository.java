package com.sap.smarthacks2024.persistence;

import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

import com.sap.smarthacks2024.model.Customer;

public interface CustomerRepository extends JpaRepository<Customer, UUID> {

}
