package com.example.backend.Repository;

import com.example.backend.Entity.parkingCar;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface parkingCarRepository extends JpaRepository<parkingCar, Integer> {
    Optional<parkingCar> findByCarNumber(String string);
    Page<parkingCar> findByCarNumberContaining(Pageable pageable, String keyword);
}