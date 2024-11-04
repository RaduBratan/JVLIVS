package com.sap.smarthacks2024.service.impl;

import java.util.Collection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.sap.smarthacks2024.model.Demand;
import com.sap.smarthacks2024.persistence.DemandRepository;
import com.sap.smarthacks2024.service.DemandService;

import jakarta.annotation.PostConstruct;

@Service
public class DemandServiceImpl implements DemandService {

	private static final Logger log = LoggerFactory.getLogger(DemandServiceImpl.class);

	private final DemandRepository demandRepository;

	private Collection<Demand> demands;

	public DemandServiceImpl(DemandRepository demandRepository) {
		this.demandRepository = demandRepository;
	}

	@Override
	public Collection<Demand> getDemands() {
		return demands;
	}

	@PostConstruct
	public void init() {
		log.atDebug().log("Initializing demands");
		demands = demandRepository.findAll();
		log.atDebug().log("{} demands initialized", demands.size());
	}

}
