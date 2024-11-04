package com.sap.smarthacks2024.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ProblemDetail;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.reactive.result.method.annotation.ResponseEntityExceptionHandler;

import com.sap.smarthacks2024.exception.BadRequestException;
import com.sap.smarthacks2024.exception.BusinessException;
import com.sap.smarthacks2024.exception.SessionAlreadyExistsException;
import com.sap.smarthacks2024.exception.SessionNotFoundException;

@RestControllerAdvice
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

	@ExceptionHandler(BadRequestException.class)
	@ResponseStatus(HttpStatus.BAD_REQUEST)
	protected ProblemDetail handleBadRequestException(BadRequestException ex) {
		return handleBusinessException(ex);
	}

	@ExceptionHandler(SessionAlreadyExistsException.class)
	@ResponseStatus(HttpStatus.CONFLICT)
	protected ProblemDetail handleSessionAlreadyExistsException(SessionAlreadyExistsException ex) {
		return handleBusinessException(ex);
	}

	@ExceptionHandler(SessionNotFoundException.class)
	@ResponseStatus(HttpStatus.NOT_FOUND)
	protected ProblemDetail handleSessionNotFoundException(SessionNotFoundException ex) {
		return handleBusinessException(ex);
	}

	@ExceptionHandler(BusinessException.class)
	@ResponseStatus
	protected ProblemDetail handleBusinessException(BusinessException ex) {
		var problemDetail = ProblemDetail.forStatusAndDetail(ex.getStatusCode(), ex.getMessage());
		problemDetail.setProperty("code", ex.getCode());
		return problemDetail;
	}
}
