package com.example.backend.Entity;

import jakarta.persistence.*;
import lombok.Data;

@Entity
@Data
@Table(name="preferentialTreatmentCar")
public class PreferentialTreatmentCar {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    @Column(name="carNumber")
    private String carNumber;
}