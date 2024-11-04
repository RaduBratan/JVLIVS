package com.sap.smarthacks2024.service.impl;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import com.sap.smarthacks2024.model.DemandStatus;
import com.sap.smarthacks2024.model.EvaluationSession;
import com.sap.smarthacks2024.model.NodeStatus;
import com.sap.smarthacks2024.persistence.DemandStatusRepository;
import com.sap.smarthacks2024.service.DemandService;
import com.sap.smarthacks2024.service.DemandStatusService;

@Service
public class DemandStatusServiceImpl implements DemandStatusService {

	
	private final DemandStatusRepository demandStatusRepository;
	private final DemandService demandService;

	DemandStatusServiceImpl(DemandService demandService, DemandStatusRepository demandStatusRepository) {
		this.demandService = demandService;
		this.demandStatusRepository = demandStatusRepository;
	}

	@Transactional(readOnly = false, rollbackFor = Throwable.class, propagation = Propagation.REQUIRED)
	@Override
	public void initDemand(EvaluationSession session, Collection<NodeStatus> nodes) {

		var nodesMap = nodes.stream().collect(Collectors.toMap(ns -> ns.getNode().getId(), ns -> ns));

		var demandList = demandService.getDemands();
		var demandStatusList = demandList.stream().map(d -> {
			var demandStatus = new DemandStatus();
			demandStatus.setDemand(d);
			demandStatus.setSession(session);
			demandStatus.setCustomerNodeStatus(nodesMap.get(d.getCustomer().getId()));
			demandStatus.setRemainingQuantity(d.getQuantity());
			return demandStatus;
		}).toList();

		demandStatusRepository.saveAll(demandStatusList);

	}

	@Override
	public List<DemandStatus> getDemands(EvaluationSession evaluationSession) {
		return demandStatusRepository.findBySessionIdAndPostDay(evaluationSession.getId(),
				evaluationSession.getCurrentDay());
	}

}
