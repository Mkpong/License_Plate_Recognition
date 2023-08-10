package com.example.backend.Repository;

import com.example.backend.Entity.PreferentialTreatmentCar;
import com.example.backend.Entity.parkingCar;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface PTCarRepository extends JpaRepository<PreferentialTreatmentCar , Integer> {
    Optional<PreferentialTreatmentCar> findByCarNumber(String string);
    Page<PreferentialTreatmentCar> findByCarNumberContaining(Pageable pageable, String keyword);
}
