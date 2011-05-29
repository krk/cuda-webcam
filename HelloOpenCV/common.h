#ifndef COMMON_H_
#define COMMON_H_

/** 
	\file common.h 
	Her sýnýfta içerilen baþlýk dosyasý.
	
	Her sýnýfta kullanýlan fonksiyonlarý ve makrolarý içeren baþlýk dosyasý.
*/

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
/** Kopyalama yaratýcýsý ve atama operatörünü private olarak iþaretler. */
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  void operator=(const TypeName&)

#endif // COMMON_H_