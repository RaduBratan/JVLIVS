package com.sap.smarthacks2024.security;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.security.config.web.server.SecurityWebFiltersOrder;
import org.springframework.security.config.web.server.ServerHttpSecurity;
import org.springframework.security.config.web.server.ServerHttpSecurity.CsrfSpec;
import org.springframework.security.config.web.server.ServerHttpSecurity.HttpBasicSpec;
import org.springframework.security.config.web.server.ServerHttpSecurity.LogoutSpec;
import org.springframework.security.web.server.SecurityWebFilterChain;
import org.springframework.security.web.server.authentication.AuthenticationWebFilter;
import org.springframework.security.web.server.context.NoOpServerSecurityContextRepository;

@Configuration
@EnableWebFluxSecurity
public class SecurityConfig {

	@Bean
	SecurityWebFilterChain securityWebFilterChain(ServerHttpSecurity http, ApiKeyAuthenticationManager authManager,
			ApiKeyAuthenticationConverter authConverter) {

		var filter = new AuthenticationWebFilter(authManager);
		filter.setServerAuthenticationConverter(authConverter);

		return http.authorizeExchange(exchanges -> exchanges
				 .pathMatchers("/api-docs/**", "/api-docs.yaml", "/swagger-ui/**","/swagger-ui.html", "webjars/swagger-ui/**").permitAll()
				.anyExchange().authenticated()).httpBasic(HttpBasicSpec::disable).formLogin(form -> form.disable())
				.csrf(CsrfSpec::disable).securityContextRepository(NoOpServerSecurityContextRepository.getInstance())
				.logout(LogoutSpec::disable).addFilterAt(filter, SecurityWebFiltersOrder.AUTHENTICATION).build();
	}

}
