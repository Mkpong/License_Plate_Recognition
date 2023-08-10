package com.example.backend.Service;

import com.example.backend.DTO.SeasonTicketDTO;
import com.example.backend.Entity.parkingCar;
import com.example.backend.Entity.seasonTicketCar;
import com.example.backend.Repository.STCarRepository;
import com.example.backend.Repository.parkingCarRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import java.time.DateTimeException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
public class SeasonService {

    @Autowired
    STCarRepository stCarRepository;
    @Autowired
    parkingCarRepository parkingcarRepository;

    public String newSeasonCar(SeasonTicketDTO st){
        Optional<seasonTicketCar> op_season = stCarRepository.findByCarNumber(st.getCar_number());
        if(op_season.isEmpty()){
            seasonTicketCar SeasonTicket = new seasonTicketCar();
            SeasonTicket.setCarNumber(st.getCar_number());
            Optional<parkingCar> op_carinfo = parkingcarRepository.findByCarNumber(st.getCar_number());
            SeasonTicket.setAuto_pay(st.isAuto_pay());
            LocalDate currentDate = LocalDate.now();
            int season_month = Integer.parseInt(st.getMonth());
            int year = currentDate.getYear();
            int month = currentDate.getMonthValue();
            int day = currentDate.getDayOfMonth();
            if(month+season_month > 12) {
                year++;
                month = month + season_month - 12;
            }
            else{
                month = month+season_month;
            }
            try{
                currentDate = LocalDate.of(year,month,day);
            }catch(DateTimeException e){
                if(month == 12){
                    month++;
                    year++;
                }
                else month++;
                day = 1;
                currentDate = LocalDate.of(year, month, day);
            }
            SeasonTicket.setValidDate(currentDate.toString());
            stCarRepository.save(SeasonTicket);
            if(!op_carinfo.isEmpty()){
                parkingCar box = op_carinfo.get();
                box.setSt_car(SeasonTicket);
                parkingcarRepository.save(box);
            }
            return "Success";
        }
        return "Fail";
    }

    public List<seasonTicketCar> getSeasonList(){
        return stCarRepository.findAll();
    }

    public Page<seasonTicketCar> searchSeasonCar(int page, String keyword){
        List<Sort.Order> sorts = new ArrayList<>();
        sorts.add(Sort.Order.asc("validDate"));
        Pageable pageable = PageRequest.of(page,10,Sort.by(sorts));
        return stCarRepository.findByCarNumberContaining(pageable,keyword);
    }


}
