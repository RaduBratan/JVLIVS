package com.sap.smarthacks2024.controller;

import java.util.UUID;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sap.smarthacks2024.controller.dto.DayRequestDto;
import com.sap.smarthacks2024.controller.dto.DayResponseDto;
import com.sap.smarthacks2024.service.SessionService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/api/v1/")
public class PlayController {

	private final SessionService sessionService;

	PlayController(SessionService sessionService) {
		this.sessionService = sessionService;
	}

	@Operation(summary = "Start a new session", description = "Start a new session for the given API key")
	@ApiResponse(responseCode = "200", description = "Session was initialized successfully")
	@ApiResponse(responseCode = "409", description = "An active session already exists for this team")
	@PostMapping("/session/start")
	public Mono<String> startSession(@RequestHeader(name = "API-KEY", required = true) UUID apiKey) {
		return Mono.just(sessionService.createSessionForApiKey(apiKey).toString());
	}

	@Operation(summary = "End the current session", description = "End the current session for the given API key")
	@ApiResponse(responseCode = "200", description = "Session was ended successfully")
	@ApiResponse(responseCode = "404", description = "No active session found for this team")
	@PostMapping("/session/end")
	public Mono<DayResponseDto> stopSession(@RequestHeader(name = "API-KEY", required = true) UUID apiKey) {
		return Mono.just(sessionService.stopSession(apiKey));
	}

	@Operation(summary = "Play a round", description = "Play a round for the current session")
	@ApiResponse(responseCode = "200", description = "Data was posted successfully")
	@ApiResponse(responseCode = "404", description = "No active session found for this team")
	@ApiResponse(responseCode = "400", description = "Invalid data provided")
	@PostMapping("/play/round")
	public Mono<DayResponseDto> playRound(@RequestHeader(name = "API-KEY") UUID apiKey,
			@RequestHeader(name = "SESSION-ID", required = true) UUID sessionId,
			@RequestBody DayRequestDto dayRequest) {

		return Mono.just(sessionService.playRound(sessionId, dayRequest.day(), dayRequest.movements()));
	}

}
