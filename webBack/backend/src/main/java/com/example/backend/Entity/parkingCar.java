package com.example.backend.Entity;

import jakarta.persistence.*;
import lombok.Data;

@Entity
@Data
@Table(name="parkingCar")
public class parkingCar {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    @Column(name="carNumber")
    private String carNumber;

    @Column(name="enterTime")
    private String entranceTime;

    @Column(name="departureTime")
    private String departureTime;

    @Column(name="plateType")
    private int plateType;

    @OneToOne
    @JoinColumn(name="PreferentialTreatmentCarId")
    private PreferentialTreatmentCar pt_car;

    @OneToOne
    @JoinColumn(name="seasonTicketCarId")
    private seasonTicketCar st_car;


}
