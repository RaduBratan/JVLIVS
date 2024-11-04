package com.sap.smarthacks2024.service;

import java.util.List;
import java.util.UUID;

import com.sap.smarthacks2024.controller.dto.DayResponseDto;
import com.sap.smarthacks2024.controller.dto.MovementDto;

public interface SessionService {
	UUID createSessionForApiKey(UUID teamId);
	
	DayResponseDto playRound(UUID sessionId, int day, List<MovementDto> movements);

	DayResponseDto stopSession(UUID apiKey);
}
