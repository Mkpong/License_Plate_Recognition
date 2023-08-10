package com.example.backend.Repository;

import com.example.backend.Entity.history;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface HistoryRepository extends JpaRepository<history, Integer> {
    Page<history> findByCarNumberContaining(Pageable pageable, String keyword);
}