package com.example.backend.Repository;

import com.example.backend.Entity.parkingCar;
import com.example.backend.Entity.seasonTicketCar;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface STCarRepository extends JpaRepository<seasonTicketCar , Integer> {
    Optional<seasonTicketCar> findByCarNumber(String string);
    Page<seasonTicketCar> findByCarNumberContaining(Pageable pageable, String keyword);
}