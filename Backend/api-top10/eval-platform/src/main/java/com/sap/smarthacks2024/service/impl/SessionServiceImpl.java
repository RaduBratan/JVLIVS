package com.sap.smarthacks2024.service.impl;

import java.text.MessageFormat;
import java.time.LocalDateTime;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import com.sap.smarthacks2024.controller.dto.DayResponseDto;
import com.sap.smarthacks2024.controller.dto.DemandDto;
import com.sap.smarthacks2024.controller.dto.KpisDto;
import com.sap.smarthacks2024.controller.dto.MovementDto;
import com.sap.smarthacks2024.controller.dto.PenaltyDto;
import com.sap.smarthacks2024.exception.BadRequestException;
import com.sap.smarthacks2024.exception.BusinessException;
import com.sap.smarthacks2024.exception.SessionAlreadyExistsException;
import com.sap.smarthacks2024.exception.SessionNotFoundException;
import com.sap.smarthacks2024.model.Demand;
import com.sap.smarthacks2024.model.EvaluationSession;
import com.sap.smarthacks2024.model.EvaluationTrack;
import com.sap.smarthacks2024.model.Penalty;
import com.sap.smarthacks2024.persistence.EvaluationTrackRepository;
import com.sap.smarthacks2024.persistence.PenaltiesRepository;
import com.sap.smarthacks2024.persistence.SessionRepository;
import com.sap.smarthacks2024.service.DemandService;
import com.sap.smarthacks2024.service.DemandStatusService;
import com.sap.smarthacks2024.service.NetworkService;
import com.sap.smarthacks2024.service.SessionService;
import com.sap.smarthacks2024.service.TeamService;

import jakarta.annotation.PostConstruct;

@Service
public class SessionServiceImpl implements SessionService {

	private final TeamService teamService;
	private final NetworkService networkService;
	private final DemandService demandService;
	private final DemandStatusService demandStatusService;
	private final SessionRepository sessionRepository;
	private final PenaltiesRepository penaltiesRepository;
	private final EvaluationTrackRepository evaluationTrackRepository;

	private Map<Integer, List<Demand>> dailyDemands;

	private final int numberOfDays;

	SessionServiceImpl(TeamService teamService, SessionRepository sessionRepository, NetworkService networkService,
			DemandService demandService, DemandStatusService demandStatusService,
			PenaltiesRepository penaltiesRepository, EvaluationTrackRepository evaluationTrackRepository,
			@Value("${game.numberOfDays}") int numberOfDays) {
		this.teamService = teamService;
		this.sessionRepository = sessionRepository;
		this.networkService = networkService;
		this.demandService = demandService;
		this.penaltiesRepository = penaltiesRepository;
		this.evaluationTrackRepository = evaluationTrackRepository;
		this.numberOfDays = numberOfDays;
		this.demandStatusService = demandStatusService;
	}

	@Transactional(readOnly = false, rollbackFor = Throwable.class, propagation = Propagation.REQUIRED)
	@Override
	public UUID createSessionForApiKey(UUID apiKey) {
		var team = teamService.getTeamByApiKey(apiKey).orElseThrow(
				() -> new BusinessException(HttpStatus.FORBIDDEN, "SESS-001", "No team found for API key"));

		if (sessionRepository.existsByTeamIdAndEndTimeIsNull(team.getId())) {
			throw new SessionAlreadyExistsException();
		}

		EvaluationSession session = new EvaluationSession();
		session.setTeam(team);
		session.setCo2(0);
		session.setCo2(0);
		session.setStartTime(LocalDateTime.now());

		session = sessionRepository.save(session);
		var nodes = networkService.initNetwork(session);
		demandStatusService.initDemand(session, nodes);
		evaluationTrackRepository.markAsNotLatest();

		return session.getId();
	}

	@Override
	@Transactional(readOnly = false, rollbackFor = Throwable.class, propagation = Propagation.REQUIRED)
	public DayResponseDto playRound(UUID sessionId, int day, List<MovementDto> movements) {
		var evaluationSession = sessionRepository.findById(sessionId).orElseThrow(SessionNotFoundException::new);
		if (evaluationSession.getCurrentDay() != day) {
			throw new BadRequestException("SESS-004", MessageFormat.format(
					"Provided day ({0}) is not the expected day ({1})", day, evaluationSession.getCurrentDay()));
		}
		var cost = evaluationSession.getCost();
		var co2 = evaluationSession.getCo2();
		var evaluationTrack = new EvaluationTrack();
		evaluationTrack.setTeam(evaluationSession.getTeam());
		evaluationTrack.setSessionId(evaluationSession.getId());
		evaluationTrack.setLatest(true);
		evaluationTrack.setTimeReceived(LocalDateTime.now());
		evaluationTrack.setProdDay(day);

		var movementsPenalties = networkService.registerMovements(evaluationSession, movements);
		List<Demand> demands = dailyDemands.get(day);
		var allPenalties = networkService.updateNetworkAndKpis(evaluationSession, movementsPenalties, evaluationTrack);
		var nextDay = day + 1;

		if (nextDay > numberOfDays) {
			evaluationSession.setEndTime(LocalDateTime.now());
			Collection<Penalty> endOfGamePenalties = endOfGame(evaluationSession, 50.0);
			var endOfGamePenaltyCost = endOfGamePenalties.stream().mapToDouble(Penalty::getCost).sum();
			var endOfGamePenaltyCo2 = endOfGamePenalties.stream().mapToDouble(Penalty::getCo2).sum();
			evaluationTrack.setPenaltyCost(evaluationTrack.getPenaltyCost() + endOfGamePenaltyCost);
			evaluationTrack.setPenaltyCo2(evaluationTrack.getPenaltyCo2() + endOfGamePenaltyCo2);
			allPenalties.addAll(endOfGamePenalties);
		} else {
			evaluationSession.setCurrentDay(nextDay);
		}
		if (!allPenalties.isEmpty()) {
			penaltiesRepository.saveAll(allPenalties);
		}

		var newCost = evaluationSession.getCost();
		var newCo2 = evaluationSession.getCo2();

		sessionRepository.save(evaluationSession);
		evaluationTrackRepository.save(evaluationTrack);

		return new DayResponseDto(day,
				demands == null ? List.of()
						: demands.stream()
								.map(d -> new DemandDto(d.getCustomer().getId(), d.getQuantity(), d.getPostDay(),
										d.getStartDeliveryDay(), d.getEndDeliveryDay()))
								.toList(),
				allPenalties.stream()
						.map(p -> new PenaltyDto(day, p.getType(), p.getMessage(), p.getCost(), p.getCo2())).toList(),
				new KpisDto(day, newCost - cost, newCo2 - co2), new KpisDto(day, newCost, newCo2));
	}

	@Override
	public DayResponseDto stopSession(UUID apiKey) {
		var evaluationSession = sessionRepository.findByApiKey(apiKey)
				.orElseThrow(() -> new SessionNotFoundException());
		evaluationSession.setEndTime(LocalDateTime.now());
		var penalties = endOfGame(evaluationSession, 700.0 * numberOfDays / evaluationSession.getCurrentDay());
		sessionRepository.save(evaluationSession);
		return new DayResponseDto(evaluationSession.getCurrentDay(), null,
				penalties.stream()
						.map(p -> new PenaltyDto(evaluationSession.getCurrentDay(), p.getType(), p.getMessage(),
								p.getCost(), p.getCo2()))
						.toList(),
				null, new KpisDto(evaluationSession.getCurrentDay(), evaluationSession.getCost(),
						evaluationSession.getCo2()));

	}

	public Collection<Penalty> endOfGame(EvaluationSession evaluationSession, double factor) {
		return networkService.endOfGame(evaluationSession, factor, numberOfDays);

	}

	@PostConstruct
	void init() {
		dailyDemands = demandService.getDemands().stream().collect(Collectors.groupingBy(Demand::getPostDay));
	}

}
