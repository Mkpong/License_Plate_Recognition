package com.example.backend.Service;

import com.example.backend.DTO.carDataDTO;
import com.example.backend.DTO.carInfoDTO;
import com.example.backend.Entity.PreferentialTreatmentCar;
import com.example.backend.Entity.parkingCar;
import com.example.backend.Entity.history;
import com.example.backend.Entity.seasonTicketCar;
import com.example.backend.Repository.PTCarRepository;
import com.example.backend.Repository.STCarRepository;
import com.example.backend.Repository.parkingCarRepository;
import com.example.backend.Repository.HistoryRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
public class ParkingCarService {

    @Autowired
    parkingCarRepository parkingcarRepository;
    @Autowired
    HistoryRepository historyService;
    @Autowired
    PTCarRepository ptCarRepository;

    @Autowired
    STCarRepository stCarRepository;

    DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public carDataDTO addCar(carInfoDTO data){
        carDataDTO car_data = new carDataDTO();
        Optional<parkingCar> op_carInfo = parkingcarRepository.findByCarNumber(data.getResult());
        if(op_carInfo.isEmpty()){
            parkingCar carinfo = new parkingCar();
            Optional<PreferentialTreatmentCar> op_ptCar = ptCarRepository.findByCarNumber(data.getResult());
            Optional<seasonTicketCar> op_stCar = stCarRepository.findByCarNumber((data.getResult()));
            if(!op_stCar.isEmpty()){
                carinfo.setSt_car(op_stCar.get());
                car_data.setTicket("정기권");
            }
            else if(!op_ptCar.isEmpty()){
                carinfo.setPt_car(op_ptCar.get());
                car_data.setTicket("우대권");
            }
            else{
                car_data.setTicket("일반");
            }
            carinfo.setCarNumber(data.getResult());
            carinfo.setEntranceTime(data.getTime());
            carinfo.setPlateType(data.getPlateType());
            parkingcarRepository.save(carinfo);
            car_data.setState("entrance");
            return car_data;
        }
        else {
            parkingCar find_parkingCar = op_carInfo.get();
            history re = new history();
            re.setCarNumber(find_parkingCar.getCarNumber());
            re.setEntranceTime(find_parkingCar.getEntranceTime());
            re.setPlateType(find_parkingCar.getPlateType());
            re.setDepartureTime(data.getTime());
            LocalDateTime dt1 = LocalDateTime.parse(find_parkingCar.getEntranceTime(), formatter);
            LocalDateTime dt2 = LocalDateTime.parse(data.getTime(), formatter);
            Duration duration = Duration.between(dt1, dt2);
            long parkingTime = duration.toSeconds();
            long days = duration.toDays();
            duration = duration.minusDays(days);
            long hours = duration.toHours();
            duration = duration.minusHours(hours);
            long minutes = duration.toMinutes();
            duration = duration.minusMinutes(minutes);
            long seconds = duration.getSeconds();
            System.out.println("Days : " + days + "hours : " +  hours + "minutes : " + minutes + "seconds: " + seconds);
            int count = 0;
            count = ((int)parkingTime / 15) * 1000;

            if (find_parkingCar.getSt_car() != null) {
                count = 0;
                car_data.setTicket("정기권");
            }
            else if (find_parkingCar.getPt_car() != null) {
                count /= 2;
                car_data.setTicket("할인권");
            }
            else{
                car_data.setTicket("일반");
            }
            car_data.setState("departure");
            car_data.setParkingFee(count);
            car_data.setParkingTime(String.format("%02d:%02d:%02d:%02d", days, hours, minutes, seconds));
            re.setParkingFee(count);
            historyService.save(re);
            parkingcarRepository.deleteById(find_parkingCar.getId());
            return car_data;
        }

    }

    public List<parkingCar> getCar(){
        return parkingcarRepository.findAll();
    }


    public Page<parkingCar> searchParkingCar(int page, String keyword){
        List<Sort.Order> sorts = new ArrayList<>();
        sorts.add(Sort.Order.desc("entranceTime"));
        Pageable pageable = PageRequest.of(page,10,Sort.by(sorts));
        return parkingcarRepository.findByCarNumberContaining(pageable,keyword);
    }

}