package com.sap.smarthacks2024.service.impl;

import java.util.Collection;

import org.springframework.stereotype.Service;

import com.sap.smarthacks2024.model.Customer;
import com.sap.smarthacks2024.model.Refinery;
import com.sap.smarthacks2024.model.StorageTank;
import com.sap.smarthacks2024.persistence.CustomerRepository;
import com.sap.smarthacks2024.persistence.RefineryRepository;
import com.sap.smarthacks2024.persistence.StorageTankRepository;
import com.sap.smarthacks2024.service.NodeService;

import jakarta.annotation.PostConstruct;

@Service
public class NodeServiceImpl implements NodeService {

	private final CustomerRepository customerRepository;
	private final RefineryRepository refineryRepository;
	private final StorageTankRepository storageTankRepository;
	private Collection<Refinery> refineries;
	private Collection<StorageTank> storageTanks;
	private Collection<Customer> customers;

	public NodeServiceImpl(CustomerRepository customerRepository, RefineryRepository refineryRepository,
			StorageTankRepository storageTankRepository) {
		this.customerRepository = customerRepository;
		this.refineryRepository = refineryRepository;
		this.storageTankRepository = storageTankRepository;
	}

	@Override
	public Collection<Refinery> getRefineries() {
		return refineries;
	}

	@Override
	public Collection<StorageTank> getStorageTanks() {
		return storageTanks;
	}

	@Override
	public Collection<Customer> getCustomers() {
		return customers;
	}

	@PostConstruct
	void init() {
		refineries = refineryRepository.findAll();
		storageTanks = storageTankRepository.findAll();
		customers = customerRepository.findAll();
	}

}
