package com.example.backend.Service;


import com.example.backend.Controller.PreferentialController;
import com.example.backend.Entity.PreferentialTreatmentCar;
import com.example.backend.Repository.PTCarRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
public class PreferentialService {

    @Autowired
    PTCarRepository ptCarRepository;

    public Page<PreferentialTreatmentCar> searchPreferentialCar(int page, String keyword){
        List<Sort.Order> sorts = new ArrayList<>();
        sorts.add(Sort.Order.desc("id"));
        Pageable pageable = PageRequest.of(page,10,Sort.by(sorts));
        return ptCarRepository.findByCarNumberContaining(pageable,keyword);
    }

    public String addPreferentialCar(String carNumber){
        Optional<PreferentialTreatmentCar> op_PtCar = ptCarRepository.findByCarNumber(carNumber);
        if(op_PtCar.isEmpty()){
            PreferentialTreatmentCar box_PtCar = new PreferentialTreatmentCar();
            box_PtCar.setCarNumber(carNumber);
            ptCarRepository.save(box_PtCar);
            return "Success";
        }
        else{
            return "Fail";
        }
    }

}
