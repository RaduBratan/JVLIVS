package com.sap.smarthacks2024.service.impl;

import java.text.MessageFormat;
import java.time.LocalDateTime;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import com.sap.smarthacks2024.controller.dto.MovementDto;
import com.sap.smarthacks2024.model.Connection;
import com.sap.smarthacks2024.model.Customer;
import com.sap.smarthacks2024.model.DemandStatus;
import com.sap.smarthacks2024.model.EvaluationSession;
import com.sap.smarthacks2024.model.EvaluationTrack;
import com.sap.smarthacks2024.model.Movement;
import com.sap.smarthacks2024.model.Node;
import com.sap.smarthacks2024.model.NodeStatus;
import com.sap.smarthacks2024.model.Penalty;
import com.sap.smarthacks2024.model.Refinery;
import com.sap.smarthacks2024.model.StorageTank;
import com.sap.smarthacks2024.persistence.ConnectionRepository;
import com.sap.smarthacks2024.persistence.DemandStatusRepository;
import com.sap.smarthacks2024.persistence.MovementRepository;
import com.sap.smarthacks2024.persistence.NodeStatusRepository;
import com.sap.smarthacks2024.service.NetworkService;
import com.sap.smarthacks2024.service.NodeService;

import jakarta.annotation.PostConstruct;

@Service
public class NetworkServiceImpl implements NetworkService {

	private static final Logger logger = LoggerFactory.getLogger(NetworkServiceImpl.class);

	private final NodeStatusRepository nodeStatusRepository;
	private final ConnectionRepository connectionRepository;
	private final MovementRepository movementRepository;
	private final DemandStatusRepository demandStatusRepository;
	private final NodeService nodeService;

	private Map<UUID, Connection> connections;

	public NetworkServiceImpl(NodeService nodeService, NodeStatusRepository nodeStatusRepository,
			ConnectionRepository connectionRepository, MovementRepository movementRepository,
			DemandStatusRepository demandStatusRepository) {

		this.nodeStatusRepository = nodeStatusRepository;
		this.connectionRepository = connectionRepository;
		this.movementRepository = movementRepository;
		this.demandStatusRepository = demandStatusRepository;
		this.nodeService = nodeService;
	}

	@Override
	@Transactional(readOnly = false, rollbackFor = Throwable.class, propagation = Propagation.REQUIRED)
	public Collection<NodeStatus> initNetwork(EvaluationSession session) {
		var refineries = nodeService.getRefineries();
		var storageTanks = nodeService.getStorageTanks();
		var customers = nodeService.getCustomers();

		var nodeStatuses = new LinkedList<NodeStatus>();

		nodeStatuses.addAll(refineries.stream().map(ref -> {
			var ns = new NodeStatus();
			ns.setNode(ref);
			ns.setSession(session);
			ns.setStock(ref.getInitialStock());
			return ns;
		}).toList());

		nodeStatuses.addAll(storageTanks.stream().map(st -> {
			var ns = new NodeStatus();
			ns.setNode(st);
			ns.setSession(session);
			ns.setStock(st.getInitialStock());
			return ns;
		}).toList());

		nodeStatuses.addAll(customers.stream().map(c -> {
			var ns = new NodeStatus();
			ns.setNode(c);
			ns.setSession(session);
			ns.setStock(0L);
			return ns;
		}).toList());

		nodeStatusRepository.saveAll(nodeStatuses);
		return nodeStatuses;

	}

	@Override
	public List<Penalty> registerMovements(EvaluationSession evaluationSession, List<MovementDto> movements) {
		if (movements == null) {
			return List.of();
		}

		var penalties = new LinkedList<Penalty>();
		var movementsToProcess = new LinkedList<Movement>();
		for (MovementDto m : movements) {
			var connection = connections.get(m.connectionId());

			if (connection != null) {
				var movement = new Movement();

				if (m.amount() < 0) {
					var penalty = new Penalty();
					var amount = m.amount();
					if (amount < 0) {
						amount = -amount * 100;
					} else if (amount == 0) {
						amount = 100;
					}
					penalty.setSession(evaluationSession);
					penalty.setDay(evaluationSession.getCurrentDay());
					penalty.setCo2(amount * connection.getDistance() * connection.getLeadTimeDays()
							* connection.getConnectionType().getCo2PerDistanceAndVolume());
					penalty.setCost(amount * connection.getDistance() * connection.getLeadTimeDays()
							* connection.getConnectionType().getCo2PerDistanceAndVolume());
					penalty.setType("INVALID_MOVEMENT_AMOUNT");
					penalty.setMessage(MessageFormat.format("Movement of {0} units was for connection {1}", m.amount(),
							m.connectionId()));
					penalties.add(penalty);
				} else {
					movement.setSession(evaluationSession);
					movement.setDayPosted(evaluationSession.getCurrentDay());
					movement.setAmount(m.amount());

					movement.setConnection(connection);
					movement.setDayDelivered(connection.getLeadTimeDays() + evaluationSession.getCurrentDay());
					movement.setFrom(nodeStatusRepository.findBySessionIdAndNodeId(evaluationSession.getId(),
							connection.getFrom().getId()));
					movement.setTo(nodeStatusRepository.findBySessionIdAndNodeId(evaluationSession.getId(),
							connection.getTo().getId()));
					movementsToProcess.add(movement);
				}
			} else {
				logger.atDebug().log("Connection not found: {}", m.connectionId());
				var penalty = new Penalty();
				var amount = m.amount();
				if (amount < 0) {
					amount = -amount * 100;
				} else if (amount == 0) {
					amount = 100;
				}
				penalty.setSession(evaluationSession);
				penalty.setDay(evaluationSession.getCurrentDay());
				penalty.setCo2(amount * 100d);
				penalty.setCost(amount * 150d);
				penalty.setType("INVALID_CONNECTION");
				penalty.setMessage(
						MessageFormat.format("Connection with id {0} does not exist. Movement of {1} was rejected",
								m.connectionId(), m.amount()));
				penalties.add(penalty);
			}

		}
		if (!movementsToProcess.isEmpty()) {
			movementRepository.saveAll(movementsToProcess);
		}
		return penalties;

	}

	@Override
	@Transactional(readOnly = false, rollbackFor = Throwable.class, propagation = Propagation.REQUIRED)
	public List<Penalty> updateNetworkAndKpis(EvaluationSession evaluationSession, List<Penalty> penalties,
			EvaluationTrack evalTrack) {

		UUID sessionId = evaluationSession.getId();
		var refineries = nodeStatusRepository.findBySessionIdAndNodeNodeType(sessionId, "REFINERY");

		double productionCost = 0d;
		double productionCo2 = 0d;

		for (var refineryStatus : refineries) {
			var refinery = (Refinery) refineryStatus.getNode();
			refineryStatus.setStock(refineryStatus.getStock() + refinery.getProduction());
			double crtCost = refinery.getProductionCostPerUnit() * refinery.getProduction();
			productionCost += crtCost;
			double crtCo2 = refinery.getProductionCO2PerUnit() * refinery.getProduction();
			productionCo2 += crtCo2;

//			refineryStatus.setCost(refineryStatus.getCost()+crtCost);
//			refineryStatus.setCo2(refineryStatus.getCo2()+crtCo2);

		}
		nodeStatusRepository.saveAll(refineries);

		Map<UUID, Long> stockDelta = new HashMap<>();

		var allPenalties = new LinkedList<Penalty>(penalties);
		var todayInMovements = movementRepository.findBySessionIdAndDayDelivered(sessionId,
				evaluationSession.getCurrentDay());
		var todayOutMovements = movementRepository.findBySessionIdAndDayPosted(sessionId,
				evaluationSession.getCurrentDay());

		var movementsCost = todayInMovements.stream().mapToDouble(m -> m.getAmount()
				* m.getConnection().getConnectionType().getCostPerDistanceAndVolume() * m.getConnection().getDistance())
				.sum();
		var movementsCo2 = todayInMovements.stream().mapToDouble(m -> m.getAmount()
				* m.getConnection().getConnectionType().getCo2PerDistanceAndVolume() * m.getConnection().getDistance())
				.sum();

		Map<UUID, List<Movement>> outMovements = todayOutMovements.stream()
				.collect(Collectors.groupingBy(m -> m.getConnection().getFrom().getId()));

		outMovements.forEach((nodeId, movements) -> {
			var nodeStatus = movements.get(0).getFrom();
			var node = nodeStatus.getNode();
			long totalAmount = movements.stream().mapToLong(Movement::getAmount).sum();
			if (node instanceof Refinery refinery && totalAmount > refinery.getMaxOutput()) {
				allPenalties.add(buildPenalty(evaluationSession, node, totalAmount - refinery.getMaxOutput(),
						refinery.getOverOutputPenaltyCoefficient(), "REFINERY_OVER_OUTPUT", "over output"));
			} else if (node instanceof StorageTank storageTank && totalAmount > storageTank.getMaxOutput()) {
				allPenalties.add(buildPenalty(evaluationSession, node, totalAmount - storageTank.getMaxOutput(),
						storageTank.getOverOutputPenaltyCoefficient(), "STORAGE_TANK_OVER_OUTPUT", "over output"));
			}

			updateStockDeltas(stockDelta, nodeStatus.getId(), -totalAmount);

		});

		Map<UUID, List<Movement>> inMovements = todayInMovements.stream()
				.collect(Collectors.groupingBy(m -> m.getConnection().getTo().getId()));

		inMovements.forEach((nodeId, movements) -> {
			var nodeStatus = movements.get(0).getTo();
			var node = nodeStatus.getNode();
			long totalAmount = movements.stream().mapToLong(Movement::getAmount).sum();
			if (node instanceof StorageTank storageTank) {
				updateStockDeltas(stockDelta, nodeStatus.getId(), totalAmount);
				if (totalAmount > storageTank.getMaxInput()) {
					allPenalties.add(buildPenalty(evaluationSession, node, totalAmount - storageTank.getMaxInput(),
							storageTank.getOverInputPenaltyCoefficient(), "STORAGE_TANK_OVER_INPUT", "over input"));
				}
			} else if (node instanceof Customer customer) {
				allPenalties.addAll(matchAndUpdateDemand(evaluationSession, nodeStatus, totalAmount));
				if (totalAmount > customer.getMaxInput()) {
					allPenalties.add(buildPenalty(evaluationSession, node, totalAmount - customer.getMaxInput(),
							customer.getOverInputPenaltyCoefficient(), "CUSTOMER_OVER_INPUT", "over input"));
				}
			}

		});

		stockDelta.forEach(nodeStatusRepository::updateStocksWithDeltas);

		allPenalties.addAll(determineStockRelatedPenalties(evaluationSession));

		var connectionsOverCapacity = connectionRepository.findConnectionsOverCapacity(sessionId,
				evaluationSession.getCurrentDay());

		allPenalties.addAll(connectionsOverCapacity.stream().map(c -> {
			var penalty = new Penalty();
			var coef = c.getConnection().getConnectionType().getOverUsePenaltyPerVolume();
			var maxCapacity = c.getConnection().getMaxCapacity();
			var penaltyValue = (c.getAmountInTransit() - maxCapacity) * coef;
			penalty.setSession(evaluationSession);
			penalty.setDay(evaluationSession.getCurrentDay());
			penalty.setCo2(penaltyValue);
			penalty.setCost(penaltyValue);
			penalty.setType("CONNECTION_OVER_CAPACITY");
			penalty.setMessage(MessageFormat.format("Connection with id {0} is over capacity. {1} units are in transit",
					c.getConnection().getId(), c.getAmountInTransit()));
			return penalty;
		}).toList());

		var penaltiesCost = allPenalties.stream().mapToDouble(Penalty::getCost).sum();
		var penaltiesCo2 = allPenalties.stream().mapToDouble(Penalty::getCo2).sum();

		var totalCost = productionCost + movementsCost + penaltiesCost;
		var totalCo2 = productionCo2 + movementsCo2 + penaltiesCo2;

		evalTrack.setMovementCost(movementsCost);
		evalTrack.setMovementCo2(movementsCo2);

		evalTrack.setProductionCost(productionCost);
		evalTrack.setProductionCo2(productionCo2);

		evalTrack.setPenaltyCo2(penaltiesCo2);
		evalTrack.setPenaltyCost(penaltiesCost);

		evaluationSession.setCost(evaluationSession.getCost() + totalCost);
		evaluationSession.setCo2(evaluationSession.getCo2() + totalCo2);
		evaluationSession.setLastUpdated(LocalDateTime.now());

		return allPenalties;
	}

	private Collection<Penalty> matchAndUpdateDemand(EvaluationSession session, NodeStatus nodeStatus,
			long totalAmount) {

		Collection<DemandStatus> demandCandidates = List.of();
		if (nodeStatus.getDemand() != null) {
			demandCandidates = nodeStatus
					.getDemand().stream().filter(d -> d.getRemainingQuantity() > 0).sorted((ds1, ds2) -> Integer
							.compare(ds1.getDemand().getStartDeliveryDay(), ds2.getDemand().getStartDeliveryDay()))
					.toList();
		}

		if (demandCandidates.isEmpty()) {
			Customer customer = (Customer) nodeStatus.getNode();
			return List.of(buildPenalty(session, customer, totalAmount,
					(customer.getEarlyDeliveryPenaltyCoefficient() + customer.getLateDeliveryPenaltyCoefficient())
							* 100,
					"CUSTOMER_UNEXPECTED_DELIVERY", "unexpected delivery"));
		}

		Collection<Penalty> penalties = new LinkedList<>();
		var demandIterator = demandCandidates.iterator();
		var currentAmount = totalAmount;
		var currentDay = session.getCurrentDay();
		var customer = (Customer) nodeStatus.getNode();
		while (currentAmount > 0 && demandIterator.hasNext()) {
			var demandStatus = demandIterator.next();
			var demand = demandStatus.getDemand();
			long deliveredValue = Math.min(currentAmount, demandStatus.getRemainingQuantity());
			if (demand.getStartDeliveryDay() > currentDay) {
				penalties.add(buildPenalty(session, customer, deliveredValue,
						customer.getEarlyDeliveryPenaltyCoefficient() * (demand.getStartDeliveryDay() - currentDay),
						"CUSTOMER_EARLY_DELIVERY", "early delivery"));
			} else if (demand.getEndDeliveryDay() < currentDay) {
				penalties.add(buildPenalty(session, customer, deliveredValue,
						customer.getLateDeliveryPenaltyCoefficient() * (currentDay - demand.getEndDeliveryDay()),
						"CUSTOMER_LATE_DELIVERY", "late delivery"));
			}
			demandStatus.setRemainingQuantity(demandStatus.getRemainingQuantity() - deliveredValue);
			currentAmount -= deliveredValue;

		}
		nodeStatusRepository.save(nodeStatus);
		if (currentAmount > 0) {

			penalties.add(buildPenalty(session, customer, currentAmount,
					(customer.getEarlyDeliveryPenaltyCoefficient() + customer.getLateDeliveryPenaltyCoefficient())
							* 100,
					"CUSTOMER_UNEXPECTED_DELIVERY", "unexpected delivery"));
		}
		return penalties;
	}

	private Penalty buildPenalty(EvaluationSession evaluationSession, Node node, long baseOverDelivery,
			double penaltyCoefficient, String penaltyType, String penaltyName) {
		var penalty = new Penalty();
		penalty.setSession(evaluationSession);
		penalty.setDay(evaluationSession.getCurrentDay());
		penalty.setCo2(baseOverDelivery * penaltyCoefficient);
		penalty.setCost(baseOverDelivery * penaltyCoefficient);
		penalty.setType(penaltyType);
		penalty.setNode(node);
		penalty.setMessage(
				MessageFormat.format("{0} affected by an {2} by {1}", node.getName(), baseOverDelivery, penaltyName));
		return penalty;
	}

	private Collection<Penalty> determineStockRelatedPenalties(EvaluationSession evaluationSession) {
		Collection<Penalty> allPenalties = new LinkedList<>();
		var sessionId = evaluationSession.getId();
		allPenalties.addAll(
				nodeStatusRepository.findBySessionIdAndStockGreaterThanCapacity(sessionId).stream().map(nodeStatus -> {
					var node = nodeStatus.getNode();
					double overflowCoef = 0d;
					long capacity = 0L;
					String penaltyType = "";
					if (node instanceof Refinery refinery) {
						overflowCoef = refinery.getOverflowPenaltyCoefficient();
						capacity = refinery.getCapacity();
						penaltyType = "REFINERY_OVERFLOW";
					} else if (node instanceof StorageTank storageTank) {
						overflowCoef = storageTank.getOverflowPenaltyCoefficient();
						capacity = storageTank.getCapacity();
						penaltyType = "STORAGE_TANK_OVERFLOW";
					}
					long overValue = Math.abs(nodeStatus.getStock() - capacity);
					return buildPenalty(evaluationSession, node, overValue, overflowCoef, penaltyType, "overflow");
				}).toList());

		allPenalties
				.addAll(nodeStatusRepository.findBySessionIdAndStockLessThanZero(sessionId).stream().map(nodeStatus -> {
					var node = nodeStatus.getNode();
					double underflowCoef = 0d;
					String penaltyType = "";
					if (node instanceof StorageTank storageTank) {
						underflowCoef = storageTank.getUnderflowPenaltyCoefficient();
						penaltyType = "STORAGE_TANK_UNDERFLOW";
					} else if (node instanceof Refinery refinery) {
						underflowCoef = refinery.getUnderflowPenaltyCoefficient();
						penaltyType = "REFINERY_UNDERFLOW";
					}
					long underValue = Math.abs(nodeStatus.getStock());
					return buildPenalty(evaluationSession, node, underValue, underflowCoef, penaltyType, "underflow");
				}).filter(penalty -> penalty.getCo2() != null && penalty.getCost() > 0 && penalty.getCo2() > 0.0
						&& penalty.getCost() > 0).toList());
		return allPenalties;
	}

	private void updateStockDeltas(Map<UUID, Long> stockDeltas, UUID nodeStatusId, long amount) {

		var crtValue = stockDeltas.getOrDefault(nodeStatusId, 0L);
		crtValue += amount;
		stockDeltas.put(nodeStatusId, crtValue);
	}

	@Override
	public NodeStatus getNoteStatusForNode(EvaluationSession evaluationSession, Node node) {
		return nodeStatusRepository.findBySessionIdAndNodeId(evaluationSession.getId(), node.getId());
	}

	@Override
	public Collection<Penalty> endOfGame(EvaluationSession evaluationSession, double factor, int numberOfDays) {
		Collection<Penalty> penalties = new LinkedList<>();
		var day = evaluationSession.getCurrentDay();
		var pendingMovements = movementRepository.findBySessionIdAndInTransit(evaluationSession.getId(), day);

		var pendingMovementsCost = pendingMovements.stream()
				.collect(Collectors.summingDouble(m -> m.getAmount() * (m.getDayDelivered() - day)
						* m.getConnection().getConnectionType().getCostPerDistanceAndVolume()
						* m.getConnection().getDistance() / m.getConnection().getLeadTimeDays()));
		var pendingMovementsCo2 = pendingMovements.stream()
				.collect(Collectors.summingDouble(m -> m.getAmount() * (m.getDayDelivered() - day)
						* m.getConnection().getConnectionType().getCo2PerDistanceAndVolume()
						* m.getConnection().getDistance() / m.getConnection().getLeadTimeDays()));

		if (pendingMovementsCost > 0 || pendingMovementsCo2 > 0) {
			var penalty = new Penalty();
			penalty.setSession(evaluationSession);
			penalty.setDay(day);
			penalty.setCost(pendingMovementsCost * factor);
			penalty.setCo2(pendingMovementsCo2 * factor);
			penalty.setType("PENDING_MOVEMENTS");
			penalty.setMessage("There are still movements in transit");
			penalties.add(penalty);
		}

		var unmetDemands = demandStatusRepository
				.findBySessionIdAndHavingRemainingQuantities(evaluationSession.getId());

		var unmetDemandBase = unmetDemands.stream()
				.collect(Collectors.summingDouble(d -> d.getRemainingQuantity()
						* (unmetDemandDays(d, day, numberOfDays)) * 10
						* ((Customer) d.getCustomerNodeStatus().getNode()).getLateDeliveryPenaltyCoefficient()));
		if (unmetDemandBase > 0) {
			var penalty = new Penalty();
			penalty.setSession(evaluationSession);
			penalty.setDay(day);
			penalty.setCost(unmetDemandBase * factor);
			penalty.setCo2(unmetDemandBase * factor);
			penalty.setType("UNMET_DEMANDS");
			penalty.setMessage("There are still unmet demands");
			penalties.add(penalty);
		}

		evaluationSession.setCost(evaluationSession.getCost() + penalties.stream().mapToDouble(Penalty::getCost).sum());
		evaluationSession.setCo2(evaluationSession.getCo2() + penalties.stream().mapToDouble(Penalty::getCo2).sum());

		return penalties;
	}

	@PostConstruct
	void init() {
		connections = connectionRepository.findAll().stream().collect(Collectors.toMap(Connection::getId, c -> c));
	}

	private int unmetDemandDays(DemandStatus ds, int currentDay, int numDays) {
		if (ds.getDemand().getQuantity() == ds.getRemainingQuantity()) {
			return (numDays - ds.getDemand().getStartDeliveryDay())*10;
		}
		return numDays - ds.getDemand().getEndDeliveryDay();
	}

}
