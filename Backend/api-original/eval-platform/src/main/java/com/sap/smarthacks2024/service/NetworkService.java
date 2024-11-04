package com.sap.smarthacks2024.service;

import java.util.Collection;
import java.util.List;

import com.sap.smarthacks2024.controller.dto.MovementDto;
import com.sap.smarthacks2024.model.EvaluationSession;
import com.sap.smarthacks2024.model.EvaluationTrack;
import com.sap.smarthacks2024.model.Node;
import com.sap.smarthacks2024.model.NodeStatus;
import com.sap.smarthacks2024.model.Penalty;

public interface NetworkService {
	Collection<NodeStatus> initNetwork(EvaluationSession session);

	List<Penalty> registerMovements(EvaluationSession evaluationSession, List<MovementDto> movements);

	List<Penalty> updateNetworkAndKpis(EvaluationSession evaluationSession, List<Penalty> penalties,
			EvaluationTrack evaluationTrack);

	NodeStatus getNoteStatusForNode(EvaluationSession evaluationSession, Node node);

	Collection<Penalty> endOfGame(EvaluationSession evaluationSession, double factor, int numberOfDays);

	

}
