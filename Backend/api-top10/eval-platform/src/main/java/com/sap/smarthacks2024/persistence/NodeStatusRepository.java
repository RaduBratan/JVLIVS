package com.sap.smarthacks2024.persistence;

import java.util.List;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;

import com.sap.smarthacks2024.model.NodeStatus;

public interface NodeStatusRepository extends JpaRepository<NodeStatus, UUID> {

	@Query("SELECT ns FROM NodeStatus ns WHERE ns.session.id = :sessionId AND ns.node.id = :nodeId")
	NodeStatus findBySessionIdAndNodeId(UUID sessionId, UUID nodeId);

	@Query("SELECT ns FROM NodeStatus ns WHERE ns.session.id = :id AND ns.node.type = :type")
	List<NodeStatus> findBySessionIdAndNodeNodeType(UUID id, String type);

	@Modifying
	@Query("UPDATE NodeStatus ns SET ns.stock = ns.stock + :deltaAmount WHERE ns.id = :nodeStatusId")
	void updateStocksWithDeltas(UUID nodeStatusId, Long deltaAmount);

	@Query("SELECT ns FROM NodeStatus ns WHERE ns.session.id = :sessionId AND ns.stock > ns.node.capacity AND ns.node.type in ('REFINERY','STORAGE_TANK')")
	List<NodeStatus> findBySessionIdAndStockGreaterThanCapacity(UUID sessionId);

	@Query("SELECT ns FROM NodeStatus ns WHERE ns.session.id = :sessionId AND ns.stock < 0 AND ns.node.type in ('REFINERY','STORAGE_TANK')")
	List<NodeStatus> findBySessionIdAndStockLessThanZero(UUID sessionId);

}
