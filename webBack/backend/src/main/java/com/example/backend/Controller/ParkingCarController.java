package com.example.backend.Controller;

import com.example.backend.DTO.carDataDTO;
import com.example.backend.DTO.carInfoDTO;
import com.example.backend.Entity.parkingCar;
import com.example.backend.Service.ParkingCarService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
public class ParkingCarController {

    @Autowired
    ParkingCarService parkingCarService;

    @PostMapping ("/detect")
    public carDataDTO addCar(@RequestBody carInfoDTO data){
        return parkingCarService.addCar(data);
    }

    @GetMapping("/carinfolist")
    public List<parkingCar> getCar(){
        return parkingCarService.getCar();
    }


    @PostMapping("/parking/state/search")
    public Page<parkingCar> searchSeasonCar(@RequestParam(value="page", defaultValue = "0") int page,
                                            @RequestParam(value="keyword") String keyword){
        return parkingCarService.searchParkingCar(page, keyword);
    }



}