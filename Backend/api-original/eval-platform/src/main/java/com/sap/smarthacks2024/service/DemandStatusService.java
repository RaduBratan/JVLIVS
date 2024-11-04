package com.sap.smarthacks2024.service;

import java.util.Collection;
import java.util.List;

import com.sap.smarthacks2024.model.DemandStatus;
import com.sap.smarthacks2024.model.EvaluationSession;
import com.sap.smarthacks2024.model.NodeStatus;

public interface DemandStatusService {
	void initDemand(EvaluationSession session, Collection<NodeStatus> nodes);

	List<DemandStatus> getDemands(EvaluationSession evaluationSession);

}
